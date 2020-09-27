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
from matplotlib.figure import Figure



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

    points = np.array(points, dtype=np.int32).reshape(17, 2)
    lleg = points[15]
    rleg = points[16]
    leye = points[11]
    reye = points[12]
    dis1 = abs(lleg[1] - leye[1])  # 左眼和左脚的y方向差距
    dis2 = abs(rleg[1] - rleg[1])
    dis3 = math.sqrt((points[11, 0] - points[12, 0]) ** 2 + (points[11, 1] - points[12, 1]) ** 2)  # 髋关节距离
    if (dis1 < 2*dis3):
        text.delete('1.0', 'end')
        text.insert(INSERT, '当前状态\n')
        text.insert(END, '跌倒了')
        text.insert(INSERT, '\n')
        #print("跌倒了！")
    else:
        text.delete('1.0', 'end')
        text.insert(INSERT, '当前状态\n')
        text.insert(END, '未跌倒')
        text.insert(INSERT, '\n')
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
    if (init_lenratio>1.5 and init_lenratio<1.9):
        text.insert(INSERT, '当前身体角度\n')
        text.insert(END, '0°')
        text.insert(INSERT, '\n')
        #print('zhongjian')
    if(init_lenratio>8):
        text.insert(INSERT, '当前身体角度\n')
        text.insert(END, '90°')
        text.insert(INSERT, '\n')
        #print('90')
    if(init_lenratio>2.3 and init_lenratio<2.6):
        text.insert(INSERT, '当前身体角度\n')
        text.insert(END, '45°')
        text.insert(INSERT, '\n')
        #print('45')

    #if(init_lenratio)
#-----------------------------------------------------------------------
    if(points[0,0]>x_c+10):
        #text.delete('1.0', 'end')
        text.insert(INSERT, '当前头部状态\n')
        text.insert(END, 'left')
        #print('left')

    elif(points[0,0]<x_c-10):
        #print('rigrht')
        #text.delete('1.0', 'end')
        text.insert(INSERT, '当前头部状态\n')
        text.insert(END, 'right')
    else:
        #text.delete('1.0', 'end')
        text.insert(INSERT, '当前头部状态\n')
        text.insert(END, 'mid')
        #print('mid')
    root.update()
#------------------------------------------------------------------------------------------
#保存关节点坐标和几个角度以及势能场
    # booksheet.write(glo, 0, float(0))
    # booksheet.write(glo, 1, float(points[5, 0] - height))
    # booksheet.write(glo, 2, float(points[5, 1] - width))
    # booksheet.write(glo, 3, float(points[6, 0] - height))
    # booksheet.write(glo, 4, float(points[6, 1] - width))
    # booksheet.write(glo, 5, float(points[7, 0] - height))
    # booksheet.write(glo, 6, float(points[7, 1] - width))
    # booksheet.write(glo, 7, float(points[8, 0] - height))
    # booksheet.write(glo, 8, float(points[8, 1] - width))
    # booksheet.write(glo, 9, float(points[9, 0] - height))
    # booksheet.write(glo, 10, float(points[9, 1] - width))
    # booksheet.write(glo, 11, float(points[10, 0] - height))
    # booksheet.write(glo, 12, float(points[10, 1] - width))
    # booksheet.write(glo, 13, float(points[11, 0] - height))
    # booksheet.write(glo, 14, float(points[11, 1] - width))
    # booksheet.write(glo, 15, float(points[12, 0] - height))
    # booksheet.write(glo, 16, float(points[12, 1] - width))
    # booksheet.write(glo, 17, float(points[13, 0] - height))
    # booksheet.write(glo, 18, float(points[13, 1] - width))
    # booksheet.write(glo, 19, float(points[14, 0] - height))
    # booksheet.write(glo, 20, float(points[14, 1] - width))
    # booksheet.write(glo, 21, float(points[15, 0] - height))
    # booksheet.write(glo, 22, float(points[15, 1] - width))
    # booksheet.write(glo, 23, float(points[16, 0] - height))
    # booksheet.write(glo, 24, float(points[16, 1] - width))
    # booksheet.write(glo, 25, float(points[0,0]))
    # booksheet.write(glo, 26, float(points[0,1]))
    # booksheet.write(glo, 27, float(points[1, 0]))
    # booksheet.write(glo, 28, float(points[1, 1]))
    # booksheet.write(glo, 29, float(points[2, 0]))
    # booksheet.write(glo, 30, float(points[2, 1]))
    # booksheet.write(glo, 31, float(points[3, 0]))
    # booksheet.write(glo, 32, float(points[3, 1]))
    # booksheet.write(glo, 33, float(points[4, 0]))
    # booksheet.write(glo, 34, float(points[4, 1]))

    # booksheet.write(glo, 35, float(A1))
    # booksheet.write(glo, 36, float(A2))
    # booksheet.write(glo, 37, float(A5))
    # booksheet.write(glo, 38, float(A6))
    # booksheet.write(glo, 39, float(A7))
    # booksheet.write(glo, 40, float(A8))
    # booksheet.write(glo, 41, float(A11))
    # booksheet.write(glo, 42, float(A12))
    # booksheet.write(glo, 43, float(A13))
    # booksheet.write(glo, 44, float(A14))

    # distance = {}
    # field = 0
    # # 势能场
    # for j in range(17):
    #     distance[j] = math.sqrt((points[j, 0] - xcenter) ** 2 + (points[j, 1] - ycenter) ** 2)
    #     booksheet.write(glo,45+j,float(distance[j]))
    #     field = field + distance[j]
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

        # #直接算的角度
    # B8=get_3dangel(points[8,0],points[8,1],d8[0],points[6,0],points[6,1],d6[0],points[10,0],points[10,1],d10[0])
    # B7=get_3dangel(points[7,0],points[7,1],d7[0],points[5,0],points[5,1],d5[0],points[9,0],points[9,1],d9[0])
    # B6=get_3dangel(points[6,0],points[6,1],d6[0],points[8,0],points[8,1],d8[0],points[12,0],points[12,1],d12[0])
    # B5=get_3dangel(points[5,0],points[5,1],d5[0],points[7,0],points[7,1],d7[0],points[11,0],points[11,1],d11[0])
    # B12=get_3dangel(points[12,0],points[12,1],d12[0],points[14,0],points[14,1],d14[0],points[11,0],points[11,1],d11[0])
    # B11=get_3dangel(points[11,0],points[11,1],d11[0],points[13,0],points[13,1],d13[0],points[12,0],points[12,1],d12[0])
    # B14=get_3dangel(points[14,0],points[14,1],d14[0],points[16,0],points[16,1],d16[0],points[12,0],points[12,1],d12[0])
    # B13=get_3dangel(points[13,0],points[13,1],d13[0],points[15,0],points[15,1],d15[0],points[11,0],points[11,1],d11[0])



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
    text.delete('1.0', 'end')

    text.insert(INSERT, '3D坐标\n')
    text.insert(END, str(P5).replace('(','').replace(')',',')+str(dee[0]))
    text.insert(INSERT, '\n')
    text.insert(END, str(P6).replace('(', '').replace(')', ',') + str(dee[0]))
    text.insert(INSERT, '\n')
    text.insert(END, str(P7).replace('(', '').replace(')', ',') + str(dee[0]))
    text.insert(INSERT, '\n')
    text.insert(END, str(P8).replace('(', '').replace(')', ',') + str(dee[0]))
    text.insert(INSERT, '\n')
    text.insert(END, str(P9).replace('(', '').replace(')', ',') + str(dee[0]))
    text.insert(INSERT, '\n')
    text.insert(END, str(P10).replace('(', '').replace(')', ',') + str(dee[0]))
    text.insert(INSERT, '\n')
    text.insert(END, str(P11).replace('(', '').replace(')', ',') + str(dee[0]))
    text.insert(INSERT, '\n')
    text.insert(END, str(P12).replace('(', '').replace(')', ',') + str(dee[0]))
    text.insert(INSERT, '\n')
    text.insert(END, str(P13).replace('(', '').replace(')', ',') + str(dee[0]))
    text.insert(INSERT, '\n')
    text.insert(END, str(P14).replace('(', '').replace(')', ',') + str(dee[0]))
    text.insert(INSERT, '\n')
    text.insert(END, str(P15).replace('(', '').replace(')', ',') + str(dee[0]))
    text.insert(INSERT, '\n')
    text.insert(END, str(P16).replace('(', '').replace(')', ',') + str(dee[0]))
    text.insert(INSERT, '\n')


    #利用深度相机的x，y，z直接算出的角度
    E8 = get_3dangel(P8[0], P8[1], dee[3], P6[0], P6[1], dee[1], P10[0], P10[1], dee[5])

    E7 = get_3dangel(P7[0], P7[1], dee[2], P5[0], P5[1], dee[0], P9[0], P9[1], dee[4])

    E6 = get_3dangel(P6[0], P6[1], dee[1], P8[0], P8[1], dee[3], P12[0], P12[1], dee[7])

    E5 = get_3dangel(P5[0], P5[1], dee[0], P7[0], P7[1], dee[2], P11[0], P11[1], dee[6])

    E12 = get_3dangel(P12[0], P12[1], dee[7], P14[0], P14[1], dee[9], P11[0], P11[1], dee[6])

    E11 = get_3dangel(P11[0], P11[1], dee[6], P13[0], P13[1], dee[8], P12[0], P12[1], dee[7])

    E14 = get_3dangel(P14[0], P14[1], dee[9], P16[0], P16[1], dee[11], P12[0], P12[1], dee[7])

    E13 = get_3dangel(P13[0], P13[1], dee[8], P15[0], P15[1], dee[10], P11[0], P11[1], dee[6])
    text.insert(INSERT, '关节点3D角度\n')
    text.insert(END, str(E8)+'\n')
    text.insert(END, str(E7)+'\n')
    text.insert(END, str(E6)+'\n')
    text.insert(END, str(E5)+'\n')
    text.insert(END, str(E12)+'\n')
    text.insert(END, str(E11)+'\n')
    text.insert(END, str(E14)+'\n')
    text.insert(END, str(E13)+'\n')




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
    ax=[]
    ay=[]
    ay1 = []
    ay2 = []
    ay3 = []
    ay4 = []
    ay5 = []
    ay6 = []
    ay7 = []
    fname = tkinter.filedialog.askopenfilename()
    camera = cv2.VideoCapture(fname)
    camera2 = cv2.VideoCapture("/home/ywj/CenterNet/src/1m90°depth20zhen.flv")
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

            ax.append(t)
            ay.append(a[0])
            ay1.append(a[1])
            ay2.append(a[2])
            ay3.append(a[3])
            ay4.append(a[4])
            ay5.append(a[5])
            ay6.append(a[6])
            ay7.append(a[7])

            f_plot.clear()
            f_plot.plot(ax, ay, label ="A8",color = "b" )
            f_plot.plot(ax, ay1, label="A7", color="g")
            f_plot.plot(ax, ay2, label="A6", color="r")
            f_plot.plot(ax, ay3, label="A5", color="grey")
            f_plot.plot(ax, ay4, label="A12", color="black")
            f_plot.plot(ax, ay5, label="A11", color="purple")
            f_plot.plot(ax, ay6, label="A14", color="y")
            f_plot.plot(ax, ay7, label="A13", color="pink")
            canvs.draw()
            cv2.waitKey(1)

            # text.delete('1.0','end')
            text.insert(INSERT, '\n')
            text.insert(INSERT,'关节角度\n')
            text.insert(END, a[0:7])
            text.insert(INSERT, '\n')
            text.insert(INSERT, '势能场\n')
            text.insert(END,a[8])
            text.insert(INSERT, '\n')
            text.insert(INSERT, '2D关键点坐标\n')
            text.insert(END,a[10])

            img = cv2.resize(img, (640, 360))
            cv2image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            current_image1 = Image.fromarray(cv2image1)
            imgtk1 = ImageTk.PhotoImage(image=current_image1)
            panel2.imgtk = imgtk1
            panel2.config(image=imgtk1)

            depth = cv2.resize(depth, (640, 360))
            cv2image2 = cv2.cvtColor(depth, cv2.COLOR_BGR2RGBA)
            current_image2 = Image.fromarray(cv2image2)
            imgtk2 = ImageTk.PhotoImage(image=current_image2)
            panel.imgtk2 = imgtk2
            panel.config(image=imgtk2)
            t=t+1
            root.after(1, video_loop)
        else:
            return
    video_loop()

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


#录制视频，获取rgb和depth图片
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

   text.delete('1.0', 'end')
   text.insert(INSERT, '3D关键点坐标\n')
   text.insert(END, all_point_3d)

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


   cv2.namedWindow("1")

   for i in range(a):
         imgw =cv2.imread("./white.jpeg")
         #cv2.resize(imgw,(1000,1000))
         for j in range(0,12):
            print(all_point_3d[i][j])
            cv2.circle(imgw, (int(all_point_3d[i][j][0]/10), int(all_point_3d[i][j][1]/10)), 5 ,  (0, 0, 0), 0)
         print("\n")
         cv2.imshow("1",imgw)
         cv2.waitKey(0)




def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
  root = Tk()
  root.title('opencv+tkinter+CenterNet')
  root.geometry("1920x1080")
  # frame1 = LabelFrame(root)
  frame1 = LabelFrame(root, text='原始视频', bg='white')
  frame2 = LabelFrame(root)
  frame3 = Frame(root)

  fff = Figure(figsize=(5, 4), dpi=100)
  f_plot = fff.add_subplot(111)
  canvs = FigureCanvasTkAgg(fff, root)
  canvs.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=0)

  panel = Label(frame1)
  panel.pack(side=TOP,fill=BOTH)
  panel2 = Label(frame1)
  panel2.pack(side=TOP)

  var = IntVar()

  root.config(cursor='arrow')
  btn = Button(frame3, text="录制深度视频",command=getdepth)
  btn.pack(fill=X,padx=10, pady=10)
  btn2 = Button(frame3, text="获取3d视频",command=huoqu3d)
  btn2.pack(fill=X,padx=10, pady=10)
  btn3 = Button(frame3, text="保存数据",command=savefile)
  btn3.pack(fill=X,padx=10, pady=10)
  btn4 = Button(frame3, text="计算3d关键点坐标",command=compute3d)
  btn4.pack(fill=X,padx=10, pady=10)
  btn5 = Checkbutton(frame3, text='是否进行补偿', variable=var, onvalue=1, offvalue=0)
  btn5.pack(fill=X, padx=10, pady=10)
  btn6 = Button(frame3, text="制作训练集",command=zhizuoxunlianji)
  btn6.pack(fill=X,padx=10, pady=10)
  btn7 = Button(frame3, text="跌到检测",command=caijishipin)
  btn7.pack(fill=X,padx=10, pady=10)

  text = Text(frame3, width=40, height=40)

  text.pack(padx=10, pady=10)
  frame1.pack(side=RIGHT)
  # frame2.pack(side=RIGHT)
  frame3.pack()
  root.mainloop()
