import xml.dom.minidom as xmldom
import os
import numpy as np
import cv2
def processed_data(i):
    i = str(i)
    zero = '0'*(5-len(i))
    xml_filepath = os.path.abspath("D:\M3FD\Annotation/"+zero+i+".xml")
    # xml_filepath=os.path.abspath("V5-13.xml")
    # 得到文件对象
    dom_obj = xmldom.parse(xml_filepath)

    # 得到元素对象
    element_obj = dom_obj.documentElement
    bndbox = element_obj.getElementsByTagName('bndbox')

    tar_box = np.zeros((len(bndbox),8))
    # print(tar_box[1])
    # print(tar_box.shape)
    xmin_data = []
    ymin_data = []
    xmax_data = []
    ymax_data = []
    for i in range(len(bndbox)):
        xmin = element_obj.getElementsByTagName('xmin')
        ymin = element_obj.getElementsByTagName('ymin')
        xmax = element_obj.getElementsByTagName('xmax')
        ymax = element_obj.getElementsByTagName('ymax')
        xmin = xmin[i].firstChild.data
        ymin = ymin[i].firstChild.data
        xmax = xmax[i].firstChild.data
        ymax = ymax[i].firstChild.data
        xmin_data.append(int(xmin))
        ymin_data.append(int(ymin))
        xmax_data.append(int(xmax))
        ymax_data.append(int(ymax))



    tar_box[:,0],tar_box[:,6] = xmin_data,xmin_data
    tar_box[:,1],tar_box[:,3] = ymin_data,ymin_data
    tar_box[:,2],tar_box[:,4] = xmax_data,xmax_data
    tar_box[:,5],tar_box[:,7] = ymax_data,ymax_data

    return tar_box

# 读取函数，用来读取文件夹中的所有函数，输入参数是文件名
def read_directory(directory_vis,directory_ir):
    path_vis = os.listdir(directory_vis)
    path_ir = os.listdir(directory_ir)
    print(path_ir[0])
    for i in range(len(path_vis)):
        img_vis = cv2.imread(directory_vis+'/'+path_vis[i])
        img_ir = cv2.imread(directory_ir+'/'+path_ir[i])
        cv2.imwrite('D:\M3FD\Vis_Ir_com/'+ str(i*2)  + '.png',img_vis)
        cv2.imwrite('D:\M3FD\Vis_Ir_com/'+ str(i*2+1)+'.png',img_ir)
