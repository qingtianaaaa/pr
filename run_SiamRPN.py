  # --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


from utils import get_subwindow_tracking


def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)#5
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32) #5*4的矩阵
    #8*8的grid img
    size = total_stride * total_stride
    count = 0

    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale #scales = [8] 所以scale=8
            hhs = hs * scale
            anchor[count, 0] = 0 #anchor的中心坐标先全设为0 在后续第46行中进行操作赋值 anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1
    #对锚点组进行广播，并设置其坐标。加上ori偏移后，xx和yy以图像中心为原点
    #score_size是RPN网络输出特征图的大小
    #因为用的是SiamRPNVOT模型 feature_in改为了256而不是512 所以最终RPN网络输出特征图为19*19
    #即score_size=19 所以一共有19*19*5=1805个anchors 下面的anchor的shape为[1805,4]
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    # print('anchor:{}'.format(anchor.shape))
    ori = - (score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)]) #xx,yy均为19*19的矩阵
    #np.tile(a,(5,1))第一个参数为Y轴扩大倍数为5，第二个为X轴扩大倍数为1便为不复制
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten() #xx和yy展开成1805*1
    # print('xx:{},yy:{}'.format(xx.shape,yy.shape))
    #为总共1805个anchor的中心点的（x，y）坐标赋值
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor

#定义跟踪器参数
class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform] 惩罚大位移 因为下一帧的搜索范围一定在上一帧proposal的附近
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)

    # 计算grid 即将搜索图分为19*19个grid
    # grid:指的对一张图像或者是featuremap进行平均地分割，但是并不一定是一个像素对应一个grid，也可能是多个像素对应一个grid。所有grid组成一个Proposal。 见图3
    total_stride = 8

    score_size = (instance_size-exemplar_size)/total_stride+1 #19
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3] #5种宽高比
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0 #惩罚项
    window_influence = 0
    lr = 0
    # adaptive change search region #
    adaptive = True

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1

#运行网络的检测分支，得到坐标回归量和得分。
def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p):
    delta, score = net(x_crop)
    '''
    torch.Tensor.permute 置换此张量的尺寸。
    torch.Tensor.contiguous 返回包含与自张量相同的数据的连续张量。如果自张量是连续的，则此函数返回自张量。
    torch.Tensor.numpy 将自张量作为 NumPy ndarray 返回。此张量和返回的 ndarray 共享相同的底层存储。自张量的变化将反映在 ndarray 中，反之亦然。
    置换delta，其形状由 N x 4k x H x W 变为4x(kx17x17)。score形状为2x(kx17x17)，并取其后一半结果
    置换score 形状为2x(kx17x17)，并取其后一半结果。
    '''
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()
    #见图1
    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)
    '''
    用cosine窗口和尺度变化penalty对剩下的proposals得分进行重新排序。在上一个策略中执行并删除了离目标较远的proposals后，
    cosine窗口用于抑制最大位移，然后增加penalty以抑制尺寸和比例的大幅变化。最后选出得分最高的前K个proposals，
    
    并用NMS选出最终的跟踪目标位置。另外，在跟踪目标得到后，通过线性插值更新目标尺寸，保持形状平稳变化。

    '''
    # size penalty 见图2
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score #pscore按一定权值叠加一个窗分布值。找出最优得分的索引

    # window float
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    #获得目标的坐标及尺寸。delta除以scale_z映射到原图
    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr
    #由预测坐标偏移得到目标中心，宽高进行滑动平均。
    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    return target_pos, target_sz, score[best_pscore_id]


def SiamRPN_init(im, target_pos, target_sz, net):
    state = dict()
    p = TrackerConfig()
    p.update(net.cfg)
    #im：H*W*C
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    # 根据目标和输入图像的大小调整搜索区域。
    if p.adaptive:
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287  # small object big search region 目标小则扩大搜索区域
        else:
            p.instance_size = 271

        p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1
    #generate_anchor 构造出以图像中心为原点，格式为[cx, cy, w, h]的锚点矩阵。 返回W*H*K个anchor即1805个anchor
    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))

    '''
    #np.mean(img, axis=(0, 1)) 是求出各个通道的平均值，shape是 (3, )
    axis=(0, 1)其实表示的是对第0和1维共同展成的二维平面进行求均值。
    '''
    #求出第一帧图像所有像素和的平均值（所有像素相加除以像素个数）
    avg_chans = np.mean(im, axis=(0, 1))
    #p.context_amount * sum(target_sz)为填充边界。wc_z和hc_z表示纹理填充后的宽高，s_z为等效边长。
    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    #填充并截取出目标
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
    # 包裹张量并记录应用于它的操作。
    z = Variable(z_crop.unsqueeze(0))
    #运行 temple 函数计算模板结果
    net.temple(z.cuda())

    if p.windowing == 'cosine':
        #np.outer与numpy.hanning搭配在一起，用来生成高斯矩阵
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)#1805 默认沿x轴复制

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state


def SiamRPN_track(state, im):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    '''
    extract scaled crops for search region x at previous target position
    在先前的目标位置提取搜索区域 x 的缩放作物
    即在前一个目标位置为搜索区域x提取缩放的截图。
    '''
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    #tracker_eval 预测出新的位置和得分。
    target_pos, target_sz, score = tracker_eval(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state
