from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from ctypes import*
import os
import cv2
import time
import xlwt
import numpy as np
import math
import time

import xlrd
from scipy import optimize
from T2D23D import *

from opts import opts
from detectors.detector_factory import detector_factory

import skimage.io as io
import tkinter.filedialog
from tkinter import *
from PIL import Image,ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.figure import Figure
import tkinter.font as tkFont


workbook = xlwt.Workbook(encoding = "utf-8")
#workbook1 = xlrd.Workbook(encoding = "utf-8")
booksheet = workbook.add_sheet('Sheet1',cell_overwrite_ok=True)
booksheet1 = workbook.add_sheet('Sheet2',cell_overwrite_ok=True)
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
global L

# 躯干12段连杆定义
#L = [400, 340, 340, 290, 290, 580, 580, 400, 500, 500, 420, 420]

def cal_ylen(a,b,depth_img):
    i = 1
    j = 0
    m=0
    while (abs(depth_img[a + i, b][0] - depth_img[a + j, b][0])< 50):
    #while (m<20):
        #print(depth_img[a + i, b][0]-depth_img[a + j, b][0])
        i += 1
        j += 1
        m+=1
    up = j
    k = 1
    l = 0
    while (abs(depth_img[a - k, b][0] - depth_img[a - l, b][0]) < 50):
        k += 1
        l += 1
    down = l
    return up,down

def cal_xlen(a,b,depth_img):
    i = 1
    j = 0
    while (abs(depth_img[a , b+i][0] - depth_img[a , b+j][0]) < 50):
        i += 1
        j += 1
    right = j
    k = 1
    l = 0
    while (abs(depth_img[a, b-k][0] - depth_img[a, b-l][0]) < 50):
        k += 1
        l += 1
    left = l
    return right,left

#计算u，v在相机坐标系的x，y值
def cal_turexy(v,u,fx,fy,cx,cy,Z):
    X = Z*(u-cx)/fx
    Y = Z*(v-cy)/fy
    return X,Y

#计算某一关节长度
def cal_lenth(x1,y1,x2,y2):
    l = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return l

def cal_lenth1(x1,y1,z1,x2,y2,z2):
    l = math.sqrt((x1-x2)**2 + (y1-y2)**2+(z1-z2)**2)
    return l

#计算3d角度
def get_3dangel(x1,y1,z1,x2,y2,z2,x3,y3,z3):
    a = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2 + (z2 - z3) ** 2)
    b = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2 + (z1 - z3) ** 2)
    c = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    if c*b==0:
      cosA=1
    else:
      cosA=(a**2-c**2-b**2)/(-2*c*b)
    if cosA < -1.0:
      cosA=-1.0
    elif cosA>1.0:
      cosA=1.0
    A=math.acos(cosA)
    deg=math.degrees(A)
    return deg

#计算2d角度
def get_angle1(y1,x1,y2,x2,y3,x3):
    a=math.sqrt((x2-x3)**2+(y2-y3)**2)
    b=math.sqrt((x1-x3)**2+(y1-y3)**2)
    c=math.sqrt((x2-x1)**2+(y2-y1)**2)
    if c*b==0:
      cosA=1
    else:
      cosA=(a**2-c**2-b**2)/(-2*c*b)
    if cosA < -1.0:
      cosA=-1.0
    elif cosA>1.0:
      cosA=1.0
    A=math.acos(cosA)
    deg=math.degrees(A)
    return deg


def duquwenjian11(booksheet, glo, points, input_img, img_id='default'):
    # 将points转换为17个xy的形式
    points = np.array(points, dtype=np.int32).reshape(17, 2)
    # 计算2d角度

    x_c = (points[6, 0] + points[5, 0]) / 2
    y_c = (points[6, 1] + points[5, 1]) / 2

    A1 = get_angle1(x_c, y_c, points[0, 0], points[0, 1], points[12, 0], points[12, 1])
    A2 = get_angle1(x_c, y_c, points[0, 0], points[0, 1], points[11, 0], points[11, 1])
    A5 = get_angle1(points[5, 0], points[5, 1], points[7, 0], points[7, 1], points[11, 0], points[11, 1])
    A6 = get_angle1(points[6, 0], points[6, 1], points[8, 0], points[8, 1], points[12, 0], points[12, 1])
    A7 = get_angle1(points[7, 0], points[7, 1], points[5, 0], points[5, 1], points[9, 0], points[9, 1])
    A8 = get_angle1(points[8, 0], points[8, 1], points[6, 0], points[6, 1], points[10, 0], points[10, 1])
    A11 = get_angle1(points[11, 0], points[11, 1], points[13, 0], points[13, 1], points[12, 0], points[12, 1])
    A12 = get_angle1(points[12, 0], points[12, 1], points[14, 0], points[14, 1], points[11, 0], points[11, 1])
    A13 = get_angle1(points[13, 0], points[13, 1], points[15, 0], points[15, 1], points[11, 0], points[11, 1])
    A14 = get_angle1(points[14, 0], points[14, 1], points[16, 0], points[16, 1], points[12, 0], points[12, 1])
    xcenter = (points[11, 0] + points[12, 0]) / 2
    ycenter = (points[11, 1] + points[12, 1]) / 2


#跌倒检测
    points = np.array(points, dtype=np.int32).reshape(17, 2)
    lleg = points[15]
    rleg = points[16]
    leye = points[11]
    reye = points[12]
    dis1 = abs(lleg[1] - leye[1])  # 左眼和左脚的y方向差距
    dis2 = abs(rleg[1] - rleg[1])
    dis3 = math.sqrt((points[11, 0] - points[12, 0]) ** 2 + (points[11, 1] - points[12, 1]) ** 2)  # 髋关节距离
    # if (dis1 < 2*dis3):
    #     text.delete('1.0', 'end')
    #     text.insert(INSERT, '当前状态\n')
    #     text.insert(END, '跌倒了')
    #     text.insert(INSERT, '\n')
    #     #print("跌倒了！")
    # else:
    #     text.delete('1.0', 'end')
    #     text.insert(INSERT, '当前状态\n')
    #     text.insert(END, '未跌倒')
    #     text.insert(INSERT, '\n')
        #print("未跌倒！")
    height = 0
    width = 0

#判断转向角度，利用身体长宽比例
    # cv2.circle(input_img, (points[5,0],points[5,1]), 10, (255,255,255), -1)
    # cv2.imshow('1',input_img)
    jiankuan = abs(points[6,0] - points[5,0])
    shenchang = abs(points[6,1]-points[12,1])
    init_lenratio = shenchang/jiankuan
    #print(init_lenratio)
#     if (init_lenratio>1.5 and init_lenratio<1.9):
#         text.insert(INSERT, '当前身体角度\n')
#         text.insert(END, '0°')
#         text.insert(INSERT, '\n')
#         #print('zhongjian')
#     if(init_lenratio>8):
#         text.insert(INSERT, '当前身体角度\n')
#         text.insert(END, '90°')
#         text.insert(INSERT, '\n')
#         #print('90')
#     if(init_lenratio>2.3 and init_lenratio<2.6):
#         text.insert(INSERT, '当前身体角度\n')
#         text.insert(END, '45°')
#         text.insert(INSERT, '\n')
#         #print('45')
#
#     #if(init_lenratio)
# #-----------------------------------------------------------------------
#     if(points[0,0]>x_c+10):
#         #text.delete('1.0', 'end')
#         text.insert(INSERT, '当前头部状态\n')
#         text.insert(END, 'left')
#         #print('left')
#
#     elif(points[0,0]<x_c-10):
#         #print('rigrht')
#         #text.delete('1.0', 'end')
#         text.insert(INSERT, '当前头部状态\n')
#         text.insert(END, 'right')
#     else:
#         #text.delete('1.0', 'end')
#         text.insert(INSERT, '当前头部状态\n')
#         text.insert(END, 'mid')
#         #print('mid')
#     root.update()


#------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------
    booksheet.write(glo, 0, float(A1))
    booksheet.write(glo, 1, float(A2))
    booksheet.write(glo, 2, float(A5))
    booksheet.write(glo, 3, float(A6))
    booksheet.write(glo, 4, float(A7))
    booksheet.write(glo, 5, float(A8))
    booksheet.write(glo, 6, float(A11))
    booksheet.write(glo, 7, float(A12))
    booksheet.write(glo, 8, float(A13))
    booksheet.write(glo, 9, float(A14))




def duquwenjian(depth_img,glo, points, input_img, img_id='default'):
    # btn7 = Checkbutton(frame3, text='是否进行补偿', variable=var, onvalue=1, offvalue=0, command=print_selection)
    # btn7.pack(fill=X, padx=10, pady=10)
    global L
    #将points转换为17个xy的形式
    points = np.array(points, dtype=np.int32).reshape(17, 2)
    #计算2d角度
    A8=get_angle1(points[8,0],points[8,1],points[6,0],points[6,1],points[10,0],points[10,1])
    A7=get_angle1(points[7,0],points[7,1],points[5,0],points[5,1],points[9,0],points[9,1])
    A6=get_angle1(points[6,0],points[6,1],points[8,0],points[8,1],points[12,0],points[12,1])
    A5=get_angle1(points[5,0],points[5,1],points[7,0],points[7,1],points[11,0],points[11,1])
    A12=get_angle1(points[12,0],points[12,1],points[14,0],points[14,1],points[11,0],points[11,1])
    A11=get_angle1(points[11,0],points[11,1],points[13,0],points[13,1],points[12,0],points[12,1])
    A14=get_angle1(points[14,0],points[14,1],points[16,0],points[16,1],points[12,0],points[12,1])
    A13=get_angle1(points[13,0],points[13,1],points[15,0],points[15,1],points[11,0],points[11,1])
    xcenter=(points[11,0]+points[12,0])/2
    ycenter=(points[11,1]+points[12,1])/2
    distance={}
    field=0

    #势能场
    for j in range(17):
      distance[j]=math.sqrt((points[j, 0]-xcenter)**2+(points[j, 1]-ycenter)**2)
      field = field +distance[j]
    print('field',field)

    #计算2D的长度作为初始关节点
    # if glo==1:
    #     len1 = cal_lenth(points[5,0],points[5,1],points[6,0],points[6,1])
    #     len2 = cal_lenth(points[6,0],points[6,1],points[8,0],points[8,1])
    #     len3 = cal_lenth(points[5,0],points[5,1],points[7,0],points[7,1])
    #     len4 = cal_lenth(points[8,0],points[8,1],points[10,0],points[10,1])
    #     len5 = cal_lenth(points[7, 0], points[7, 1], points[9, 0], points[9, 1])
    #     len6 = cal_lenth(points[12,0],points[12,1],points[6,0],points[6,1])
    #     len7 = cal_lenth(points[5,0],points[5,1],points[11,0],points[11,1])
    #     len8 = cal_lenth(points[11,0],points[11,1],points[12,0],points[12,1])
    #     len9 = cal_lenth(points[12,0],points[12,1],points[14,0],points[14,1])
    #     len10 = cal_lenth(points[11, 0], points[11, 1], points[13, 0], points[13, 1])
    #     len11 = cal_lenth(points[14,0],points[14,1],points[16,0],points[16,1])
    #     len12 = cal_lenth(points[15,0],points[15,1],points[13,0],points[13,1])
    #
    #     L=[len1, len2,len3,len4,len5,len6,len7,len8,len9,len10,len11,len12]
    #     print("L",L)

    #每个关节的宽度



#读取深度图上每个关节点的灰度值
    d5 = depth_img[points[5,1],points[5,0]]
    d6 = depth_img[points[6, 1], points[6, 0]]
    d7 = depth_img[points[7, 1], points[7, 0]]
    d8 = depth_img[points[8, 1], points[8, 0]]
    d9 = depth_img[points[9, 1], points[9, 0]]
    d10 = depth_img[points[10, 1], points[10, 0]]
    d11 = depth_img[points[11, 1], points[11, 0]]
    d12 = depth_img[points[12, 1], points[12, 0]]
    d13 = depth_img[points[13, 1], points[13, 0]]
    d14 = depth_img[points[14, 1], points[14, 0]]
    d15 = depth_img[points[15, 1], points[15, 0]]
    d16 = depth_img[points[16, 1], points[16, 0]]
    # print("DEPTH5-16",d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16)

   


    #平均深度
    de=[d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16]
    dee = []
    for dk in de:
        #print("dk",dk)
        #dee.append(float((dk[0]+dk[1]+dk[2])/3)*16)
        d=dk[0]*16
        dee.append(d)
    # print("dee",dee)


    if glo==1:

        # 计算了7,8(近似相等，两个肘关节)关节点手臂的直径，是用像素来表示的
        up,down=cal_ylen(points[7,1], points[7,0], depth_img)
        print("up", up)
        print("手肘down", down)
        zhijing = up + down
        banjing = zhijing / 2
        print("直径", zhijing)

        #计算手腕（左右手腕近似相等）
        up1,down1=cal_ylen(points[10,1], points[10,0], depth_img)
        print("手腕up", up1)
        print("手腕down", down1)
        zhijing1 = up1 + down1
        banjing1 = zhijing1 / 2
        print("手腕直径", zhijing1)

        #计算膝关节宽度
        right, left = cal_xlen(points[14, 1], points[14, 0], depth_img)
        print("膝盖lfet", left)
        print("膝盖right", right)
        zhijing2 = left + right
        banjing2 = zhijing2 / 2
        print("膝盖直径", zhijing2)

        #计算脚踝
        right1, left1 = cal_xlen(points[16, 1], points[16, 0], depth_img)
        print("脚踝lfet", left1)
        print("脚踝right", right1)
        zhijing3 = left1 + right1
        banjing3 = zhijing3 / 2
        print("脚踝直径", zhijing3)

        #身体厚度
        laaa = (points[5,0]-points[6,0]+points[11,0]-points[12,0])/2
        houdu = laaa/4
        hd = houdu/2

        #用来补偿的点总共是5,6,7,8,9,10,11,12,13,14,15,16
        buchang = [hd,hd,banjing,banjing,banjing1,banjing1,hd,hd,banjing2,banjing2,banjing3,banjing3]
        print(buchang)


    fx=1.0239889463911715e+03
    cx=9.0796074418257501e+02
    fy=1.0206766157435967e+03
    cy=5.5670778195232526e+02


#计算深度相机实际的xy值
    P5=cal_turexy(points[5,1],points[5,0],fx,fy,cx,cy,dee[0])
    P6=cal_turexy(points[6,1],points[6,0],fx,fy,cx,cy,dee[1])
    P7=cal_turexy(points[7,1],points[7,0],fx,fy,cx,cy,dee[2])
    P8=cal_turexy(points[8,1],points[8,0],fx,fy,cx,cy,dee[3])
    P9=cal_turexy(points[9,1],points[9,0],fx,fy,cx,cy,dee[4])
    P10=cal_turexy(points[10,1],points[10,0],fx,fy,cx,cy,dee[5])
    P11=cal_turexy(points[11,1],points[11,0],fx,fy,cx,cy,dee[6])
    P12=cal_turexy(points[12,1],points[12,0],fx,fy,cx,cy,dee[7])
    P13=cal_turexy(points[13,1],points[13,0],fx,fy,cx,cy,dee[8])
    P14=cal_turexy(points[14,1],points[14,0],fx,fy,cx,cy,dee[9])
    P15=cal_turexy(points[15,1],points[15,0],fx,fy,cx,cy,dee[10])
    P16=cal_turexy(points[16,1],points[16,0],fx,fy,cx,cy,dee[11])

    if glo == 1:
        len1 = cal_lenth1(P5[0], P5[1], dee[0],P6[0], P6[1],dee[1])
        len2 = cal_lenth1(P6[0], P6[1],dee[1], P8[0], P8[1],dee[3])
        len3 = cal_lenth1(P5[0], P5[1],dee[0], P7[0], P7[1],dee[2])
        len4 = cal_lenth1(P8[0], P8[1],dee[3], P10[0], P10[1],dee[5])
        len5 = cal_lenth1(P7[0], P7[1],dee[2], P9[0], P9[1],dee[4])
        len6 = cal_lenth1(P12[0], P12[1],dee[7], P6[0], P6[1],dee[1])
        len7 = cal_lenth1(P5[0], P5[1], dee[0],P11[0], P11[1],dee[6])
        len8 = cal_lenth1(P11[0], P11[1], dee[6],P12[0], P12[1],dee[7])
        len9 = cal_lenth1(P12[0], P12[1],dee[7], P14[0], P14[1],dee[9])
        len10 = cal_lenth1(P11[0], P11[1], dee[6],P13[0], P13[1],dee[8])
        len11 = cal_lenth1(P14[0], P14[1], dee[9],P16[0], P16[1],dee[11])
        len12 = cal_lenth1(P15[0], P15[1], dee[10],P13[0], P13[1],dee[8])

        L = [len1, len2, len3, len4, len5, len6, len7, len8, len9, len10, len11, len12]
        print("L", L)


    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(P5,dee[0])
    print(P6,dee[1])
    print(P7,dee[2])
    print(P8,dee[3])
    print(P9,dee[4])
    print(P10,dee[5])
    print(P11,dee[6])
    print(P12,dee[7])
    print(P13,dee[8])
    print(P14,dee[9])
    print(P15,dee[10])
    print(P16,dee[11])
    print('\n')

    text0.delete('1.0', 'end')
    text01.delete('1.0', 'end')
    text02.delete('1.0', 'end')

    text1.delete('1.0', 'end')
    text11.delete('1.0', 'end')
    text12.delete('1.0', 'end')

    text2.delete('1.0', 'end')
    text21.delete('1.0', 'end')
    text22.delete('1.0', 'end')

    text3.delete('1.0', 'end')
    text31.delete('1.0', 'end')
    text32.delete('1.0', 'end')

    text4.delete('1.0', 'end')
    text41.delete('1.0', 'end')
    text42.delete('1.0', 'end')

    text5.delete('1.0', 'end')
    text51.delete('1.0', 'end')
    text52.delete('1.0', 'end')

    text6.delete('1.0', 'end')
    text61.delete('1.0', 'end')
    text62.delete('1.0', 'end')

    text7.delete('1.0', 'end')
    text71.delete('1.0', 'end')
    text72.delete('1.0', 'end')

    text8.delete('1.0', 'end')
    text81.delete('1.0', 'end')
    text82.delete('1.0', 'end')

    text9.delete('1.0', 'end')
    text91.delete('1.0', 'end')
    text92.delete('1.0', 'end')

    text10.delete('1.0', 'end')
    text101.delete('1.0', 'end')
    text102.delete('1.0', 'end')

    text11x.delete('1.0', 'end')
    text111.delete('1.0', 'end')
    text112.delete('1.0', 'end')


    text0.insert(END,str(int(P5[0])))
    text01.insert(END,str(int(P5[1])))
    text02.insert(END,str(dee[0]))

    text1.insert(END,str(int(P6[0])))
    text11.insert(END,str(int(P6[1])))
    text12.insert(END,str(dee[1]))

    text2.insert(END,str(int(P7[0])))
    text21.insert(END,str(int(P7[1])))
    text22.insert(END,str(dee[2]))

    text3.insert(END,str(int(P8[0])))
    text31.insert(END,str(int(P8[1])))
    text32.insert(END,str(dee[3]))

    text4.insert(END,str(int(P9[0])))
    text41.insert(END,str(int(P9[1])))
    text42.insert(END,str(dee[4]))

    text5.insert(END,str(int(P10[0])))
    text51.insert(END,str(int(P10[1])))
    text52.insert(END,str(dee[5]))

    text6.insert(END,str(int(P11[0])))
    text61.insert(END,str(int(P11[1])))
    text62.insert(END,str(dee[6]))

    text7.insert(END,str(int(P12[0])))
    text71.insert(END,str(int(P12[1])))
    text72.insert(END,str(dee[7]))

    text8.insert(END,str(int(P13[0])))
    text81.insert(END,str(int(P13[1])))
    text82.insert(END,str(dee[8]))

    text9.insert(END,str(int(P14[0])))
    text91.insert(END,str(int(P14[1])))
    text92.insert(END,str(dee[9]))

    text10.insert(END,str(int(P15[0])))
    text101.insert(END,str(int(P15[1])))
    text102.insert(END,str(dee[10]))

    text11x.insert(END,str(int(P16[0])))
    text111.insert(END,str(int(P16[1])))
    text112.insert(END,str(dee[11]))



    #利用深度相机的x，y，z直接算出的角度
    E8 = get_3dangel(P8[0], P8[1], dee[3], P6[0], P6[1], dee[1], P10[0], P10[1], dee[5])

    E7 = get_3dangel(P7[0], P7[1], dee[2], P5[0], P5[1], dee[0], P9[0], P9[1], dee[4])

    E6 = get_3dangel(P6[0], P6[1], dee[1], P8[0], P8[1], dee[3], P12[0], P12[1], dee[7])

    E5 = get_3dangel(P5[0], P5[1], dee[0], P7[0], P7[1], dee[2], P11[0], P11[1], dee[6])

    E12 = get_3dangel(P12[0], P12[1], dee[7], P14[0], P14[1], dee[9], P11[0], P11[1], dee[6])

    E11 = get_3dangel(P11[0], P11[1], dee[6], P13[0], P13[1], dee[8], P12[0], P12[1], dee[7])

    E14 = get_3dangel(P14[0], P14[1], dee[9], P16[0], P16[1], dee[11], P12[0], P12[1], dee[7])

    E13 = get_3dangel(P13[0], P13[1], dee[8], P15[0], P15[1], dee[10], P11[0], P11[1], dee[6])



    print("**********E5",E5)
    print("**********E6", E6)
    print("**********E7", E7)
    print("**********E8", E8)
    print("**********E11", E11)
    print("**********E12", E12)
    print("**********E13", E13)
    print("**********E14", E14)
    print('\n')

    height = 0
    width = 0

#关节点xy坐标和深度z

    booksheet.write(glo, 0, float(0))
    booksheet.write(glo,1,float(points[5,0]-height))
    booksheet.write(glo,2,float(points[5,1]-width))
    booksheet.write(glo,3,float(points[6,0]-height))
    booksheet.write(glo,4,float(points[6,1]-width))
    booksheet.write(glo,5,float(points[7,0]-height))
    booksheet.write(glo,6,float(points[7,1]-width))
    booksheet.write(glo,7,float(points[8,0]-height))
    booksheet.write(glo,8,float(points[8,1]-width))
    booksheet.write(glo,9,float(points[9,0]-height))
    booksheet.write(glo,10,float(points[9,1]-width))
    booksheet.write(glo,11,float(points[10,0]-height))
    booksheet.write(glo,12,float(points[10,1]-width))
    booksheet.write(glo,13,float(points[11,0]-height))
    booksheet.write(glo,14,float(points[11,1]-width))
    booksheet.write(glo,15,float(points[12,0]-height))
    booksheet.write(glo,16,float(points[12,1]-width))
    booksheet.write(glo,17,float(points[13,0]-height))
    booksheet.write(glo,18,float(points[13,1]-width))
    booksheet.write(glo,19,float(points[14,0]-height))
    booksheet.write(glo,20,float(points[14,1]-width))
    booksheet.write(glo,21,float(points[15,0]-height))
    booksheet.write(glo,22,float(points[15,1]-width))
    booksheet.write(glo,23,float(points[16,0]-height))
    booksheet.write(glo,24,float(points[16,1]-width))
    booksheet.write(glo,25,float(dee[0]))
    booksheet.write(glo,26,float(dee[1]))
    booksheet.write(glo,27,float(dee[2]))
    booksheet.write(glo,28,float(dee[3]))
    booksheet.write(glo,29,float(dee[4]))
    booksheet.write(glo,30,float(dee[5]))
    booksheet.write(glo,31,float(dee[6]))
    booksheet.write(glo,32,float(dee[7]))
    booksheet.write(glo,33,float(dee[8]))
    booksheet.write(glo,34,float(dee[9]))
    booksheet.write(glo,35,float(dee[10]))
    booksheet.write(glo,36,float(dee[11]))

    if A6<10 and A5<10 and A12>80 and A12<100 and A11>80 and A11<100:
      booksheet.write(glo,37,float(1))
    else:
      booksheet.write(glo,37,float(0))

    booksheet1.write(glo,0,float(E5))
    booksheet1.write(glo,1,float(E6))
    booksheet1.write(glo,2,float(E7))
    booksheet1.write(glo,3,float(E8))
    booksheet1.write(glo,4,float(E11))
    booksheet1.write(glo,5,float(E12))
    booksheet1.write(glo,6,float(E13))
    booksheet1.write(glo,7,float(E14))


    return A8,A7,A6,A5,A12,A11,A14,A13,distance,field,points

def huoqu3d():
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    global t
    t=1
    plt.ion()
    axx=[]
    ayy=[]
    ayy1 = []
    ayy2 = []
    ayy3 = []
    ayy4 = []
    ayy5 = []
    ayy6 = []
    ayy7 = []
    fname = tkinter.filedialog.askopenfilename(title='Select RGB video')
    fname2 = tkinter.filedialog.askopenfilename(title='Select DEPTH video')
    camera = cv2.VideoCapture(fname)
    # camera2 = cv2.VideoCapture("/home/ywj/CenterNet/src/1m90°depth20zhen.flv")
    camera2 = cv2.VideoCapture(fname2)
    def video_loop():
        global t
        _,depth = camera2.read()

        success, img = camera.read()

        if success:
            input_img=img
            #print(input_img.shape)
            ret = detector.run(input_img)
            results=ret['results']
            #bbox存了坐标点
            for bbox in results[1]:
                if bbox[4] > opt.vis_thresh:
                    a = duquwenjian(depth,t, bbox[5:39], input_img, img_id='multi_pose')
            input_img = np.expand_dims(input_img, 0)

            axx.append(t)
            ayy.append(a[0])
            ayy1.append(a[1])
            ayy2.append(a[2])
            ayy3.append(a[3])
            ayy4.append(a[4])
            ayy5.append(a[5])
            ayy6.append(a[6])
            ayy7.append(a[7])

            f_plot.clear()
            if var8.get()==0:
                f_plot.plot(axx, ayy, label ="A8",color = "LightBlue" )
            # f_plot.plot(axx, ayy, label="A8", color="b")
            if var7.get() == 0:
                f_plot.plot(axx, ayy1, label="A7", color="g")
            if var6.get() == 0:
                f_plot.plot(axx, ayy2, label="A6", color="r")
            if var5.get()==0:
                f_plot.plot(axx, ayy3, label="A5", color="grey")
            if var12.get()==0:
                f_plot.plot(axx, ayy4, label="A12", color="orange")
            if var11.get()==0:
                f_plot.plot(axx, ayy5, label="A11", color="purple")
            if var14.get()==0:
                f_plot.plot(axx, ayy6, label="A14", color="y")
            if var13.get()==0:
                f_plot.plot(axx, ayy7, label="A13", color="pink")

            x_major_locator = plt.MultipleLocator(5)
            f_plot.xaxis.set_major_locator(x_major_locator)

            canvs.draw()
            cv2.waitKey(1)


            img = cv2.resize(img, (640, 360))
            cv2image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            current_image1 = Image.fromarray(cv2image1)
            imgtk1 = ImageTk.PhotoImage(image=current_image1)
            panel.imgtk = imgtk1
            panel.config(image=imgtk1)

            depth = cv2.resize(depth, (640, 360))
            cv2image2 = cv2.cvtColor(depth, cv2.COLOR_BGR2RGBA)
            current_image2 = Image.fromarray(cv2image2)
            imgtk2 = ImageTk.PhotoImage(image=current_image2)
            panel2.imgtk2 = imgtk2
            panel2.config(image=imgtk2)
            t=t+1
            root.after(1, video_loop)
        else:
            return
    video_loop()

##用来采集视频的函数，没调用
def caijishipin():
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    T = 0
    workbook2 = xlwt.Workbook(encoding="utf-8")
    # workbook1 = xlrd.Workbook(encoding = "utf-8")
    booksheet2 = workbook2.add_sheet('Sheet1', cell_overwrite_ok=True)
    #fname = './videodata/a04_s01_e01_rgb.avi'
    fname = tkinter.filedialog.askopenfilename()
    camera = cv2.VideoCapture(fname)
    while camera.isOpened():
        success, img = camera.read()

        if success:
            input_img = img
            # print(input_img.shape)
            ret = detector.run(input_img)
            print('T', T)
            results = ret['results']
            # bbox存了坐标点
            for bbox in results[1]:
                # if bbox[4] > opt.vis_thresh:
                if bbox[4] > 0.5:
                    a = duquwenjian11(booksheet2, T, bbox[5:39], input_img, img_id='multi_pose')
            input_img = np.expand_dims(input_img, 0)
            cv2.waitKey(1)

            T = T + 1
        else:
            break
    workbook2.save('./匹配匹配/' + '4角度' + '.xls')


def zhizuoxunlianji():
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    global t

    for f1 in range(7):
        for f2 in range(1,25):
            for f3 in range(2):
                T = 0
                workbook2 = xlwt.Workbook(encoding="utf-8")
                # workbook1 = xlrd.Workbook(encoding = "utf-8")
                booksheet2 = workbook2.add_sheet('Sheet1', cell_overwrite_ok=True)
                if f2<10:
                    fname = './videodata/a0'+str(f1)+'_'+'s0'+str(f2)+'_'+'e0'+str(f3)+'_'+'rgb.avi'
                else:
                    fname = './videodata/a0' + str(f1) + '_' + 's' + str(f2) + '_' + 'e0' + str(f3) + '_' + 'rgb.avi'
                camera = cv2.VideoCapture(fname)
                while camera.isOpened():
                    success, img = camera.read()

                    if success:
                        input_img=img
                        #print(input_img.shape)
                        ret = detector.run(input_img)
                        print('T',T)
                        results=ret['results']
                        #bbox存了坐标点
                        for bbox in results[1]:
                            #if bbox[4] > opt.vis_thresh:
                            if bbox[4]>0.8:
                                a = duquwenjian11(booksheet2,T, bbox[5:39], input_img, img_id='multi_pose')
                        input_img = np.expand_dims(input_img, 0)
                        cv2.waitKey(1)
                        T=T+1
                    else:break
                workbook2.save('./xlss/'+str(f1)+'-'+str(2*(f2-1)+f3)+'.xls')


#录制视频，获取rgb和depth图片这里是20帧得
def getdepth():
   depthcap = cdll.LoadLibrary('/home/ywj/Desktop/ren/build/KinectOneStream.so')
   depthcap.main()
   videoWriter = cv2.VideoWriter('movergb.flv', cv2.VideoWriter_fourcc(*'FLV1'), 10, (960, 540))

   for i in range(1, 20):
       # load pictures from your path
       img = cv2.imread('/home/ywj/CenterNet/src/' + str(i) + 'rgb' + '.jpg')
       img = cv2.resize(img, (960, 540))
       videoWriter.write(img)

   videoWriter.release()

   videoWriter1 = cv2.VideoWriter('movedep.flv', cv2.VideoWriter_fourcc(*'FLV1'), 10, (960, 540))

   for i in range(1, 20):
       # load pictures from your path
       img = cv2.imread('/home/ywj/CenterNet/src/' + str(i) + 'rgb2depth' + '.jpg')
       img = cv2.resize(img, (960, 540))
       videoWriter1.write(img)

   videoWriter1.release()

#自适应的
   # videoWriter = cv2.VideoWriter('123.flv', cv2.VideoWriter_fourcc(*'FLV1'), 10, (960,540))

   # path="/home/ywj/CenterNet/src"
   # dirs=os.listdir(path)
   # videoWriter = cv2.VideoWriter('111.flv', cv2.VideoWriter_fourcc(*'flv1'), 10, (960,540))
   # for i in dirs:
   #     if i.endswith('rgb.jpg'):
   #        # load pictures from your path
   #        #print("11111111",path + '/' +str(i) )
   #        img = cv2.imread(path + '/' +str(i) )
   #        img = cv2.resize(img, (960, 540))
   #        videoWriter.write(img)
   # videoWriter.release()
   # #print("down1")
   #
   #
   # # change your path; 30 is fps; (2304,1296) is screen size
   # videoWriter1 = cv2.VideoWriter('333.flv', cv2.VideoWriter_fourcc(*'flv1'), 10, (960,540))
   # for i in dirs:
   #     if i.endswith('rgb2depth.jpg'):
   #     # load pictures from your path
   #        img = cv2.imread(path + '/' + str(i))
   #        img = cv2.resize(img, (960, 540))
   #        videoWriter1.write(img)
   # videoWriter1.release()
   # #print("down2")

def savefile():
   workbook.save("1234567.xls")


#迭代法算3D
def compute3d():


   global L
   workbook1 = xlrd.open_workbook("1234567.xls")
   worksheet1 = workbook1.sheet_by_index(0)

   col = worksheet1.col_values(1)
   col.remove(col[0])
   length = len(col)

   s = []
   for i in range(length):
      point = worksheet1.row_values(i + 1)
      point.remove(point[0])
      s_target = get_s(point, L)
      s_cur = max(optimize.brent(f, args=(point, s_target, L)), s_target)
      s.append(s_cur)

#    多帧优化（无窗口滑动）
#    ll = 20
#    i_num = length // ll
#    result_s = []
#    for i in range(i_num):
#       init_s = s[i * ll:i * ll + ll]
#       begin = i * ll + 1
#       end = i * ll + ll
#       result_s_cur = optimize.fmin_powell(f_s, init_s, args=(begin, end,worksheet1, L))
#       result_s.extend(result_s_cur)
#    init_s = s[i_num * ll:length]
#    begin = i_num * ll + 1
#    end = length
#    result_s_cur = optimize.fmin_powell(f_s, init_s, args=(begin, end,worksheet1, L))
#    result_s.extend(result_s_cur)

#   多帧优化（窗口滑动）
   ll = 20
   result_s = []
   result_s.extend(s[0:20])
   for i in range(20, len(s)):
       init_s = s[i-19, i+1]
       begin = i - 18
       end = i + 1
       result_s_cur = optimize.fmin_powell(f_s, init_s, args=(begin, end,worksheet1, L))
       temp = result_s_cur[-1]
       result_s.append(temp)

   # result_s=s
   all_point_3d = []
   for i in range(length-1):
       point = worksheet1.row_values(i + 1)
       point.remove(point[0])
       s_target = result_s[i]
       all_point_3d.append(get_point_3d(point, s_target, L))
   #


   a=len(all_point_3d)
   for i in range(a):
       point = worksheet1.row_values(i + 1)
       point.remove(point[0])

       #xy用迭代法，z用深度算的角度
       lshoulder = get_3dangel(all_point_3d[i][6][0], all_point_3d[i][6][1], point[25],
                               all_point_3d[i][2][0], all_point_3d[i][2][1],
                               point[31], all_point_3d[i][7][0], all_point_3d[i][7][1],
                               point[27])
       rshoulder = get_3dangel(all_point_3d[i][9][0], all_point_3d[i][9][1],  point[24],
                               all_point_3d[i][3][0],
                               all_point_3d[i][3][1],  point[30], all_point_3d[i][10][0],
                               all_point_3d[i][10][1],
                               point[26])
       lelbow = get_3dangel(all_point_3d[i][7][0], all_point_3d[i][7][1],  point[27], all_point_3d[i][8][0],
                            all_point_3d[i][8][1], point[29], all_point_3d[i][6][0], all_point_3d[i][6][1],
                            point[25])
       relbow = get_3dangel(all_point_3d[i][10][0], all_point_3d[i][10][1], point[26],
                            all_point_3d[i][9][0],
                            all_point_3d[i][9][1], point[24], all_point_3d[i][11][0],
                            all_point_3d[i][11][1],
                            point[28])

       point = worksheet1.row_values(i + 1)
       point.remove(point[0])


       #2d角度
       lshoulder2d = get_angle1(point[2],point[3],point[6],point[7],point[14],point[15])
       rshoulder2d = get_angle1(point[0],point[1],point[12],point[13],point[4],point[5])
       lelbow2d = get_angle1(point[6],point[7],point[2],point[3],point[10],point[11])
       relbow2d = get_angle1(point[4],point[5],point[0],point[1],point[8],point[9])


       #利用迭代法算的角度
       lshoulder1=get_3dangel(all_point_3d[i][6][0],all_point_3d[i][6][1],all_point_3d[i][6][2],all_point_3d[i][2][0],all_point_3d[i][2][1],
                             all_point_3d[i][2][2],all_point_3d[i][7][0],all_point_3d[i][7][1],all_point_3d[i][7][2])
       rshoulder1 = get_3dangel(all_point_3d[i][9][0], all_point_3d[i][9][1], all_point_3d[i][9][2], all_point_3d[i][3][0],
                             all_point_3d[i][3][1], all_point_3d[i][3][2], all_point_3d[i][10][0], all_point_3d[i][10][1],
                             all_point_3d[i][10][2])
       lelbow1 = get_3dangel(all_point_3d[i][7][0], all_point_3d[i][7][1], all_point_3d[i][7][2], all_point_3d[i][8][0],
                           all_point_3d[i][8][1], all_point_3d[i][8][2], all_point_3d[i][6][0], all_point_3d[i][6][1],
                           all_point_3d[i][6][2])
       relbow1 = get_3dangel(all_point_3d[i][10][0], all_point_3d[i][10][1], all_point_3d[i][10][2], all_point_3d[i][9][0],
                             all_point_3d[i][9][1], all_point_3d[i][9][2], all_point_3d[i][11][0], all_point_3d[i][11][1],
                             all_point_3d[i][11][2])



       print("lshoulder", lshoulder)
       print("lshoulder2d", lshoulder2d)
       # print("zlshoulder1",lshoulder1)

       print("rshoulder", rshoulder)
       print("rshoulder2d", rshoulder2d)
       # print("zrshoulder1",rshoulder1)

       print("lelbow", lelbow)
       print("lelbow2d", lelbow2d)
       # print("zlelbow1",lelbow1)

       print("relbow", relbow)
       print("relbow2d", relbow2d)
       # print("zrelbow1",relbow1)

       print('\n')


   # cv2.namedWindow("1")

#迭代结果画图 并且显示坐标
   for i in range(a):
         # imgw =cv2.imread("./white.jpeg")
         fxx.clear()
         x = []
         y = []
         z = []
         for j in range(0,12):

            print(all_point_3d[i][j])
            # cv2.circle(imgw, (int(all_point_3d[i][j][0]/10), int(all_point_3d[i][j][1]/10)), 5 ,  (0, 0, 0), 0)
            x.append(int(all_point_3d[i][j][0] / 10))
            y.append(int(all_point_3d[i][j][1] / 10))
            z.append(int(all_point_3d[i][j][2] / 10))

         textt0.delete('1.0', 'end')
         textt01.delete('1.0', 'end')
         textt02.delete('1.0', 'end')

         textt1.delete('1.0', 'end')
         textt11.delete('1.0', 'end')
         textt12.delete('1.0', 'end')

         textt2.delete('1.0', 'end')
         textt21.delete('1.0', 'end')
         textt22.delete('1.0', 'end')

         textt3.delete('1.0', 'end')
         textt31.delete('1.0', 'end')
         textt32.delete('1.0', 'end')

         textt4.delete('1.0', 'end')
         textt41.delete('1.0', 'end')
         textt42.delete('1.0', 'end')

         textt5.delete('1.0', 'end')
         textt51.delete('1.0', 'end')
         textt52.delete('1.0', 'end')

         textt6.delete('1.0', 'end')
         textt61.delete('1.0', 'end')
         textt62.delete('1.0', 'end')

         textt7.delete('1.0', 'end')
         textt71.delete('1.0', 'end')
         textt72.delete('1.0', 'end')

         textt8.delete('1.0', 'end')
         textt81.delete('1.0', 'end')
         textt82.delete('1.0', 'end')

         textt9.delete('1.0', 'end')
         textt91.delete('1.0', 'end')
         textt92.delete('1.0', 'end')

         textt10.delete('1.0', 'end')
         textt101.delete('1.0', 'end')
         textt102.delete('1.0', 'end')

         textt11x.delete('1.0', 'end')
         textt111.delete('1.0', 'end')
         textt112.delete('1.0', 'end')

         textt0.insert(END, str(int(all_point_3d[i][0][0])))
         textt01.insert(END, str(int(all_point_3d[i][0][1])))
         textt02.insert(END, str(int(all_point_3d[i][0][2])))

         textt1.insert(END, str(int(all_point_3d[i][0][1])))
         textt11.insert(END, str(int(int(all_point_3d[i][1][1]))))
         textt12.insert(END, str(int(int(all_point_3d[i][1][2]))))

         textt2.insert(END, str(int(all_point_3d[i][2][0])))
         textt21.insert(END, str(int(all_point_3d[i][2][1])))
         textt22.insert(END, str(int(all_point_3d[i][2][2])))

         textt3.insert(END, str(int(all_point_3d[i][3][0])))
         textt31.insert(END, str(int(all_point_3d[i][3][1])))
         textt32.insert(END,str(int(all_point_3d[i][3][2])))

         textt4.insert(END, str(int(all_point_3d[i][4][0])))
         textt41.insert(END, str(int(all_point_3d[i][4][1])))
         textt42.insert(END, str(int(all_point_3d[i][4][2])))

         textt5.insert(END, str(int(all_point_3d[i][5][0])))
         textt51.insert(END, str(int(all_point_3d[i][5][1])))
         textt52.insert(END, str(int(all_point_3d[i][5][2])))

         textt6.insert(END, str(int(all_point_3d[i][6][0])))
         textt61.insert(END, str(int(all_point_3d[i][6][1])))
         textt62.insert(END, str(int(all_point_3d[i][6][2])))

         textt7.insert(END, str(int(all_point_3d[i][7][0])))
         textt71.insert(END, str(int(all_point_3d[i][7][1])))
         textt72.insert(END, str(int(all_point_3d[i][7][2])))

         textt8.insert(END, str(int(all_point_3d[i][8][0])))
         textt81.insert(END, str(int(all_point_3d[i][8][1])))
         textt82.insert(END, str(int(all_point_3d[i][8][2])))

         textt9.insert(END, str(int(all_point_3d[i][9][0])))
         textt91.insert(END, str(int(all_point_3d[i][9][1])))
         textt92.insert(END, str(int(all_point_3d[i][9][2])))

         textt10.insert(END, str(int(all_point_3d[i][10][0])))
         textt101.insert(END, str(int(all_point_3d[i][10][1])))
         textt102.insert(END, str(int(all_point_3d[i][10][2])))

         textt11x.insert(END, str(int(all_point_3d[i][11][0])))
         textt111.insert(END, str(int(all_point_3d[i][11][1])))
         textt112.insert(END, str(int(all_point_3d[i][11][2])))

         fxx.scatter(x, y, z, c='r')
         fxx.plot(x[0:6],y[0:6],z[0:6])
         fxx.plot(x[6:9],y[6:9],z[6:9])
         fxx.plot(x[9:12],y[9:12],z[9:12])
         xx=[x[2],x[3]]
         yy=[y[2],y[3]]
         zz=[z[2],z[3]]
         fxx.plot(xx,yy,zz)
         xx1=[x[2],x[6]]
         yy1=[y[2],y[6]]
         zz1=[z[2],z[6]]
         fxx.plot(xx1, yy1, zz1)
         xx2=[x[9],x[6]]
         yy2=[y[9],y[6]]
         zz2=[z[9],z[6]]
         fxx.plot(xx2, yy2, zz2)
         xx3=[x[9],x[3]]
         yy3=[y[9],y[3]]
         zz3=[z[9],z[3]]
         fxx.plot(xx3, yy3, zz3)

         canvs1.draw()
         print("\n")
         time.sleep(0.5)




def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
  root = Tk()
  root.title('人体3D坐标与角度测算系统')
  # root.geometry("800x800")
  root.geometry("1600x1000")
  root.resizable(0, 0)
  ft = tkFont.Font(size=10, slant=tkFont.ITALIC)


#在菜单栏放几个启动按钮---------------------------------------
  menubar = tkinter.Menu(root)
  filemenu = tkinter.Menu(menubar, tearoff=0)
  menubar.add_cascade(label='录制深度视频', menu=filemenu,font=ft)
  filemenu.add_command(label='开始录制',command=getdepth,font=ft)

  editmenu = tkinter.Menu(menubar, tearoff=0)
  menubar.add_cascade(label='获取3d视频', menu=editmenu,font=ft)
  editmenu.add_command(label='选择视频',command=huoqu3d,font=ft)

  datamenu = tkinter.Menu(menubar, tearoff=0)
  menubar.add_cascade(label='保存数据', menu=datamenu,font=ft)
  datamenu.add_command(label='确认保存', command=savefile,font=ft)

  copmenu = tkinter.Menu(menubar, tearoff=0)
  menubar.add_cascade(label='计算3d坐标', menu=copmenu,font=ft)
  copmenu.add_command(label='开始', command=compute3d,font=ft)
  var0 = IntVar()
  copmenu.add_checkbutton(label='进行补偿',font=ft)
#-----------------------------------------------------------------

  frame1=LabelFrame(root,width=640, height=400, text='原始视频',bg='white',font=ft)
  panel = Label(frame1,bitmap="gray50",width=640, height=450, bg='white',font=ft)
  panel.grid(row=0, column=0)


  frame2 = LabelFrame(root,width=640, height=400, text='深度视频',bg='white',font=ft)
  panel2 = Label(frame2,bitmap="gray50",width=640, height=450,  bg='white',font=ft)
  panel2.grid(row=0, column=0)


  frame3 = LabelFrame(root,text='3D坐标点',font=ft)

  frame5 = Frame(frame3,width=40)
  frame5.grid(column=4,rowspan=11)


  name=Label(frame3,text='左肩',font=ft)
  name.grid(row=1,column=0)
  name1=Label(frame3,text='右肩',font=ft)
  name1.grid(row=2,column=0)
  name2=Label(frame3,text='左肘',font=ft)
  name2.grid(row=3,column=0)
  name3=Label(frame3,text='右肘',font=ft)
  name3.grid(row=4,column=0)
  name4=Label(frame3,text='左手腕',font=ft)
  name4.grid(row=5,column=0)
  name5=Label(frame3,text='右手腕',font=ft)
  name5.grid(row=6,column=0)
  name6=Label(frame3,text='左髋',font=ft)
  name6.grid(row=7,column=0)
  name7=Label(frame3,text='右髋',font=ft)
  name7.grid(row=8,column=0)
  name8=Label(frame3,text='左膝盖',font=ft)
  name8.grid(row=9,column=0)
  name9=Label(frame3,text='右膝盖',font=ft)
  name9.grid(row=10,column=0)
  name10=Label(frame3,text='左脚踝',font=ft)
  name10.grid(row=11,column=0)
  name11=Label(frame3,text='右脚踝',font=ft)
  name11.grid(row=12,column=0)

  name = Label(frame3, text='X',font=ft)
  name.grid(row=0, column=1)
  name = Label(frame3, text='Y',font=ft)
  name.grid(row=0, column=2)
  name = Label(frame3, text='Z',font=ft)
  name.grid(row=0, column=3)

  #---------------------------------------
  #x
  text0 = Text(frame3,width=5,height=3)
  text0.grid(row=1, column=1)
  text1 = Text(frame3,width=5,height=3)
  text1.grid(row=2, column=1)
  text2 = Text(frame3,width=5,height=3)
  text2.grid(row=3, column=1)
  text3 = Text(frame3,width=5,height=3)
  text3.grid(row=4, column=1)
  text4 = Text(frame3,width=5,height=3)
  text4.grid(row=5, column=1)
  text5 = Text(frame3,width=5,height=3)
  text5.grid(row=6, column=1)
  text6 = Text(frame3,width=5,height=3)
  text6.grid(row=7, column=1)
  text7 = Text(frame3,width=5,height=3)
  text7.grid(row=8, column=1)
  text8 = Text(frame3,width=5,height=3)
  text8.grid(row=9, column=1)
  text9 = Text(frame3,width=5,height=3)
  text9.grid(row=10, column=1)
  text10 = Text(frame3,width=5,height=3)
  text10.grid(row=11, column=1)
  text11 = Text(frame3,width=5,height=3)
  text11.grid(row=12, column=1)

  #--------------------------------------
  #y
  text01 = Text(frame3, width=5, height=3)
  text01.grid(row=1, column=2)
  text11x = Text(frame3, width=5, height=3)
  text11x.grid(row=2, column=2)
  text21 = Text(frame3, width=5, height=3)
  text21.grid(row=3, column=2)
  text31 = Text(frame3, width=5, height=3)
  text31.grid(row=4, column=2)
  text41 = Text(frame3, width=5, height=3)
  text41.grid(row=5, column=2)
  text51 = Text(frame3, width=5, height=3)
  text51.grid(row=6, column=2)
  text61 = Text(frame3, width=5, height=3)
  text61.grid(row=7, column=2)
  text71 = Text(frame3, width=5, height=3)
  text71.grid(row=8, column=2)
  text81 = Text(frame3, width=5, height=3)
  text81.grid(row=9, column=2)
  text91 = Text(frame3, width=5, height=3)
  text91.grid(row=10, column=2)
  text101 = Text(frame3, width=5, height=3)
  text101.grid(row=11, column=2)
  text111 = Text(frame3, width=5, height=3)
  text111.grid(row=12, column=2)

  #----------------------------------------
  text02 = Text(frame3, width=5, height=3)
  text02.grid(row=1, column=3)
  text12 = Text(frame3, width=5, height=3)
  text12.grid(row=2, column=3)
  text22 = Text(frame3, width=5, height=3)
  text22.grid(row=3, column=3)
  text32 = Text(frame3, width=5, height=3)
  text32.grid(row=4, column=3)
  text42 = Text(frame3, width=5, height=3)
  text42.grid(row=5, column=3)
  text52 = Text(frame3, width=5, height=3)
  text52.grid(row=6, column=3)
  text62 = Text(frame3, width=5, height=3)
  text62.grid(row=7, column=3)
  text72 = Text(frame3, width=5, height=3)
  text72.grid(row=8, column=3)
  text82 = Text(frame3, width=5, height=3)
  text82.grid(row=9, column=3)
  text92 = Text(frame3, width=5, height=3)
  text92.grid(row=10, column=3)
  text102 = Text(frame3, width=5, height=3)
  text102.grid(row=11, column=3)
  text112 = Text(frame3, width=5, height=3)
  text112.grid(row=12, column=3)


  var8 = IntVar()
  var7 = IntVar()
  var6 = IntVar()
  var5 = IntVar()
  var12 = IntVar()
  var11 = IntVar()
  var14 = IntVar()
  var13 = IntVar()

  btn = Checkbutton(frame3, text='右肘角度', variable=var8, onvalue=1, offvalue=0, bg='LightBlue',font=ft)
  btn.grid(row=1, column=5)
  btn1 = Checkbutton(frame3, text='左肘角度', variable=var7, onvalue=1, offvalue=0,bg='green',font=ft)
  btn1.grid(row=2, column=5)
  btn2 = Checkbutton(frame3, text='右肩角度', variable=var6, onvalue=1, offvalue=0,bg='red',font=ft)
  btn2.grid(row=3, column=5)
  btn3 = Checkbutton(frame3, text='左肩角度', variable=var5, onvalue=1, offvalue=0,bg='grey',font=ft)
  btn3.grid(row=4, column=5)
  btn4 = Checkbutton(frame3, text='右髋角度', variable=var12, onvalue=1, offvalue=0,bg='orange',font=ft)
  btn4.grid(row=5, column=5)
  btn5 = Checkbutton(frame3, text='左髋角度', variable=var11, onvalue=1, offvalue=0,bg='purple',font=ft)
  btn5.grid(row=6, column=5)
  btn6 = Checkbutton(frame3, text='右膝角度', variable=var14, onvalue=1, offvalue=0,bg='yellow',font=ft)
  btn6.grid(row=7, column=5)
  btn7 = Checkbutton(frame3, text='左膝角度', variable=var13, onvalue=1, offvalue=0,bg='pink',font=ft)
  btn7.grid(row=8, column=5)


  # fff = Figure()
  fff=plt.figure()
  f_plot = fff.add_subplot(111)
  plt.title("Angle Curves")
  plt.xlabel("Frames")
  plt.ylabel("Angles/°")
  plt.xlim(0, )


  canvs = tkinter.Canvas(root, bg='green')
  canvs = FigureCanvasTkAgg(fff, root)

  frame4=LabelFrame(root, text='迭代优化',font=ft)
  frame4.grid(row=1,column=1)



  textt0 = Text(frame4,width=5,height=3)
  textt0.grid(row=1, column=1)
  textt1 = Text(frame4,width=5,height=3)
  textt1.grid(row=2, column=1)
  textt2 = Text(frame4,width=5,height=3)
  textt2.grid(row=3, column=1)
  textt3 = Text(frame4,width=5,height=3)
  textt3.grid(row=4, column=1)
  textt4 = Text(frame4,width=5,height=3)
  textt4.grid(row=5, column=1)
  textt5 = Text(frame4,width=5,height=3)
  textt5.grid(row=6, column=1)
  textt6 = Text(frame4,width=5,height=3)
  textt6.grid(row=7, column=1)
  textt7 = Text(frame4,width=5,height=3)
  textt7.grid(row=8, column=1)
  textt8 = Text(frame4,width=5,height=3)
  textt8.grid(row=9, column=1)
  textt9 = Text(frame4,width=5,height=3)
  textt9.grid(row=10, column=1)
  textt10 = Text(frame4,width=5,height=3)
  textt10.grid(row=11, column=1)
  textt11 = Text(frame4,width=5,height=3)
  textt11.grid(row=12, column=1)

  #--------------------------------------
  #y
  textt01 = Text(frame4, width=5, height=3)
  textt01.grid(row=1, column=2)
  textt11x = Text(frame4, width=5, height=3)
  textt11x.grid(row=2, column=2)
  textt21 = Text(frame4, width=5, height=3)
  textt21.grid(row=3, column=2)
  textt31 = Text(frame4, width=5, height=3)
  textt31.grid(row=4, column=2)
  textt41 = Text(frame4, width=5, height=3)
  textt41.grid(row=5, column=2)
  textt51 = Text(frame4, width=5, height=3)
  textt51.grid(row=6, column=2)
  textt61 = Text(frame4, width=5, height=3)
  textt61.grid(row=7, column=2)
  textt71 = Text(frame4, width=5, height=3)
  textt71.grid(row=8, column=2)
  textt81 = Text(frame4, width=5, height=3)
  textt81.grid(row=9, column=2)
  textt91 = Text(frame4, width=5, height=3)
  textt91.grid(row=10, column=2)
  textt101 = Text(frame4, width=5, height=3)
  textt101.grid(row=11, column=2)
  textt111 = Text(frame4, width=5, height=3)
  textt111.grid(row=12, column=2)

  #----------------------------------------
  textt02 = Text(frame4, width=5, height=3)
  textt02.grid(row=1, column=3)
  textt12 = Text(frame4, width=5, height=3)
  textt12.grid(row=2, column=3)
  textt22 = Text(frame4, width=5, height=3)
  textt22.grid(row=3, column=3)
  textt32 = Text(frame4, width=5, height=3)
  textt32.grid(row=4, column=3)
  textt42 = Text(frame4, width=5, height=3)
  textt42.grid(row=5, column=3)
  textt52 = Text(frame4, width=5, height=3)
  textt52.grid(row=6, column=3)
  textt62 = Text(frame4, width=5, height=3)
  textt62.grid(row=7, column=3)
  textt72 = Text(frame4, width=5, height=3)
  textt72.grid(row=8, column=3)
  textt82 = Text(frame4, width=5, height=3)
  textt82.grid(row=9, column=3)
  textt92 = Text(frame4, width=5, height=3)
  textt92.grid(row=10, column=3)
  textt102 = Text(frame4, width=5, height=3)
  textt102.grid(row=11, column=3)
  textt112 = Text(frame4, width=5, height=3)
  textt112.grid(row=12, column=3)

  name = Label(frame4, text='右脚', font=ft)
  name.grid(row=12, column=0)
  name1 = Label(frame4, text='右膝盖', font=ft)
  name1.grid(row=10, column=0)
  name2 = Label(frame4, text='右髋', font=ft)
  name2.grid(row=8, column=0)
  name3 = Label(frame4, text='左髋', font=ft)
  name3.grid(row=7, column=0)
  name4 = Label(frame4, text='左膝盖', font=ft)
  name4.grid(row=9, column=0)
  name5 = Label(frame4, text='左脚', font=ft)
  name5.grid(row=11, column=0)
  name6 = Label(frame4, text='右肩', font=ft)
  name6.grid(row=2, column=0)
  name7 = Label(frame4, text='右肘', font=ft)
  name7.grid(row=4, column=0)
  name8 = Label(frame4, text='右手', font=ft)
  name8.grid(row=6, column=0)
  name9 = Label(frame4, text='左肩', font=ft)
  name9.grid(row=1, column=0)
  name10 = Label(frame4, text='左肘', font=ft)
  name10.grid(row=3, column=0)
  name11 = Label(frame4, text='左手', font=ft)
  name11.grid(row=5, column=0)


  fig=plt.figure()
  fxx=Axes3D(fig)
  fxx.set_zlabel("Z")
  fxx.set_xlabel("X")
  fxx.set_ylabel("Y")
  fxx.set_title("3D model")


  canvs1 = tkinter.Canvas(root, bg='green')
  canvs1 = FigureCanvasTkAgg(fig, root)
  canvs1.get_tk_widget().grid(row=1,column=2,sticky=N)

  # root.config(cursor='arrow')

  frame1.grid(row=0,column=0,sticky=N)
  frame2.grid(row=1,column=0,sticky=N)
  frame3.grid(row=0,column=1,sticky=N)

  frame6 = Frame(frame4,width=60)
  frame6.grid(column=4,rowspan=11)
  frame7 = Frame(frame4,width=110)
  frame7.grid(column=0,rowspan=11)
  canvs.get_tk_widget().grid(row=0,column= 2,sticky=N)
  root.config(menu=menubar)
  root.mainloop()
