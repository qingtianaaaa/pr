# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNvot
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect
from xmlprocess_people import processed_data

# print(torch.cuda.is_available()) 可以使用gpu运算
# load net
net = SiamRPNvot()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))
net.eval().cuda() #将CPU上的神经网络模型放到GPU上
image_file_vis = sorted(glob.glob('D:\M3FD\Vis/*.png'))
image_file_ir = sorted(glob.glob('D:\M3FD\Ir/*.png'))
# init_rbox = processed_data(1) #返回第i个xml文件中的所有匹配目标
#得到初始bounding box(标定框) 只需要给出标定框，后续的搜索范围往往在上一帧图像的附近。
# [cx, cy, w, h] = get_axis_aligned_bbox(init_rbox[10])
# tracker init
# target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
# im = cv2.imread(image_files[0])
# state = SiamRPN_init(im, target_pos, target_sz, net)

all_iou = 0
#读取第一帧图像
num = 0
for f,image_vis in enumerate(image_file_vis):

    init_rbox = processed_data(f)
    if init_rbox is False:
        continue
    num = num + 1
    [cx, cy, w, h] = get_axis_aligned_bbox(init_rbox[0])
    x_left = round(cx - 0.5*w)
    y_left = round(cy - 0.5*h)
    x_right = round(cx+0.5*w)
    y_right = round(cy+0.5*h)
    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
    im_ir = cv2.imread(image_file_ir[f])
    im_vis = cv2.imread(image_vis)
    state_vis = SiamRPN_init(im_vis, target_pos, target_sz, net)
    state_vis = SiamRPN_track(state_vis, im_ir)

    # im_ir = cv2.imread(image_file_ir[f])
    # state_ir = SiamRPN_init(im_ir, target_pos, target_sz, net)
    # state_ir = SiamRPN_track(state_ir, im_ir)

    #Ground Truth
    res_vis = cxy_wh_2_rect(state_vis['target_pos'], state_vis['target_sz'])
    res_vis = [int(l) for l in res_vis]
    cv2.rectangle(im_ir, (res_vis[0], res_vis[1]), (res_vis[0] + res_vis[2], res_vis[1] + res_vis[3]),color = (0, 255, 255),thickness =2 )
    cv2.putText(im_ir,'GT',(res_vis[0],res_vis[1]),cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 255),2 )
    # Bounding Box
    # res_ir = cxy_wh_2_rect(state_ir['target_pos'],state_ir['target_sz'])
    # res_ir = [int(i) for i in res_ir]
    # cv2.rectangle(im_ir,(res_ir[0], res_ir[1]), (res_ir[0] + res_ir[2], res_ir[1] + res_ir[3]),color = (255,0,255),thickness = 2)
    # cv2.putText(im_ir, 'BB', (res_ir[0]+res_ir[2], res_ir[1]),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)
    cv2.rectangle(im_ir, (x_left, y_left), (x_right, y_right), color=(255, 0, 255),thickness=2)
    cv2.putText(im_ir, 'BB', (x_left+x_right, y_left), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
    #计算IOU
    x1min, y1min, x1max, y1max = res_vis[0], res_vis[1], res_vis[0] + res_vis[2], res_vis[1] + res_vis[3]
    x2min, y2min, x2max, y2max = x_left,y_left,x_right,y_right
    # 计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交部分的坐标
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection
    # 计算iou
    iou = intersection / union
    iou = ('%.2f' % iou)
    all_iou = all_iou + float(iou)
    #显示IOU
    cv2.putText(im_ir,str(iou), (res_vis[0], res_vis[1]+res_vis[3]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
    cv2.imshow('SiamRPN',im_ir)
    cv2.waitKey(1)
print(num)
print("avr_iou:{}".format(all_iou/num))