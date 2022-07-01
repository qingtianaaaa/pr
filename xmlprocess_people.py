import xml.dom.minidom as xmldom
import os
import numpy as np
import cv2

def processed_data(i):
    i = str(i)
    zero = '0'*(5-len(i))
    xml_filepath = os.path.abspath("D:\M3FD\Annotation/"+zero+i+".xml")
    # xml_filepath=os.path.abspath("D:\M3FD\Annotation/00004.xml")
    # 得到文件对象
    dom_obj = xmldom.parse(xml_filepath)

    # 得到元素对象
    element_obj = dom_obj.documentElement
    obj = element_obj.getElementsByTagName('object')

    xmin_data = []
    ymin_data = []
    xmax_data = []
    ymax_data = []
    people_num = 0
    # obj_name = element_obj.getElementsByTagName('name')
    for i in range(len(obj)):
        obj_name = element_obj.getElementsByTagName('name')
        obj_name = obj_name[i].firstChild.data
        if obj_name == 'People':
            people_num = people_num + 1
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

    if people_num ==0:
        return False
    tar_box = np.zeros((people_num,8))
    tar_box[:,0],tar_box[:,6] = xmin_data,xmin_data
    tar_box[:,1],tar_box[:,3] = ymin_data,ymin_data
    tar_box[:,2],tar_box[:,4] = xmax_data,xmax_data
    tar_box[:,5],tar_box[:,7] = ymax_data,ymax_data

    return tar_box
# a=obj_name[0].firstChild.data
# print(a=='People')
# print(len(obj))

