# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
'''
RPN 包括以下部分：

生成 anchor boxes
判断每个 anchor box 为 foreground(包含物体) 或者 background(背景) ，二分类
边界框回归(bounding box regression) 对 anchor box 进行微调，使得 positive anchor 和真实框(Ground Truth Box)更加接近
'''
class SiamRPN(nn.Module):
    def __init__(self, size=2, feature_out=512, anchor=5):
        configs = [3, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: 3 if x==3 else x*size, configs))
        #configs=[3, 192, 512, 768, 768, 512]
        feat_in = configs[-1] #512
        super(SiamRPN, self).__init__()
        self.featureExtract = nn.Sequential(
            nn.Conv2d(configs[0], configs[1] , kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )

        self.anchor = anchor #5
        self.feature_out = feature_out #512

        self.conv_r1 = nn.Conv2d(feat_in, feature_out*4*anchor,3)#坐标回归
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out*2*anchor, 3)#分类 前景or背景
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
        self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)

        self.r1_kernel = []
        self.cls1_kernel = []
        self.cfg = {}

    def forward(self, x):
        x_f = self.featureExtract(x) #([1, 256, 24, 24])

        # print('回归:{}'.format(self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel)).shape))
        # print('分类:{}'.format( F.conv2d(self.conv_cls2(x_f), self.cls1_kernel).shape))
        # print('r1_kernel:{}'.format(self.r1_kernel.shape))
        # print('cls1_kernel:{}'.format(self.cls1_kernel.shape))

        return self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel)), \
               F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)

    #处理模板
    def temple(self, z):
        z_f = self.featureExtract(z)
        # print('z_f:{}'.format(z_f.shape)) #[1,256,6,6]
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = r1_kernel_raw.data.size()[-1] #4
        self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)


class SiamRPNBIG(SiamRPN):
    def __init__(self):
        super(SiamRPNBIG, self).__init__(size=2)
        self.cfg = {'lr':0.295, 'window_influence': 0.42, 'penalty_k': 0.055, 'instance_size': 271, 'adaptive': True} # 0.383


class SiamRPNvot(SiamRPN):
    def __init__(self):
        super(SiamRPNvot, self).__init__(size=1, feature_out=256)
        self.cfg = {'lr':1, 'window_influence': 0.0, 'penalty_k': 0.0, 'instance_size': 271, 'adaptive': False} # 0.355


class SiamRPNotb(SiamRPN):
    def __init__(self):
        super(SiamRPNotb, self).__init__(size=1, feature_out=256)
        self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22, 'instance_size': 271, 'adaptive': False} # 0.655
