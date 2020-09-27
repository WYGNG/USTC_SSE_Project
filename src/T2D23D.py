import numpy as np
import matplotlib.pyplot as plt
import xlrd
import math
from scipy import optimize

# 计算角度,(x1, y1, z1)为顶点
def get_angle1(x1,y1,z1,x2,y2,z2,x3,y3,z3):
    a=math.sqrt((x2-x3)**2+(y2-y3)**2+(z2-z3)**2)
    b=math.sqrt((x1-x3)**2+(y1-y3)**2+(z1-z3)**2)
    c=math.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
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

# 躯干12段连杆定义
# L = [40, 34, 34, 29, 29, 58, 58, 40, 50, 50, 42, 42]
# 通过关节点坐标计算比例系数的初值
def get_s(point,L):
    s = []
    s.append(math.sqrt((point[0] - point[2]) ** 2 + (point[1] - point[3]) ** 2) / L[0])
    s.append(math.sqrt((point[2] - point[6]) ** 2 + (point[3] - point[7]) ** 2) / L[1])
    s.append(math.sqrt((point[0] - point[4]) ** 2 + (point[1] - point[5]) ** 2) / L[2])
    s.append(math.sqrt((point[6] - point[10]) ** 2 + (point[7] - point[11]) ** 2) / L[3])
    s.append(math.sqrt((point[4] - point[8]) ** 2 + (point[5] - point[9]) ** 2) / L[4])
    s.append(math.sqrt((point[2] - point[14]) ** 2 + (point[3] - point[15]) ** 2) / L[5])
    s.append(math.sqrt((point[0] - point[12]) ** 2 + (point[1] - point[13]) ** 2) / L[6])
    s.append(math.sqrt((point[12] - point[14]) ** 2 + (point[13] - point[15]) ** 2) / L[7])
    s.append(math.sqrt((point[14] - point[18]) ** 2 + (point[15] - point[19]) ** 2) / L[8])
    s.append(math.sqrt((point[12] - point[16]) ** 2 + (point[13] - point[17]) ** 2) / L[9])
    s.append(math.sqrt((point[18] - point[22]) ** 2 + (point[19] - point[23]) ** 2) / L[10])
    s.append(math.sqrt((point[16] - point[20]) ** 2 + (point[17] - point[21]) ** 2) / L[11])
    s_target = max(s)
    #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",s_target)
    return s_target

#由2D关节点坐标和比例系数s计算3D关节点坐标
def get_point_3d(point, s_target,L):
    z0 = 525 / s_target
    point_3d = []

    point_3d.append([point[22] / s_target, point[23] / s_target, z0]) # 0
    dz11 = math.sqrt(
        max(L[10] ** 2 - ((point[18] - point[22]) ** 2 + (point[19] - point[23]) ** 2) / (s_target ** 2), 0))
    if point[33]<point[35]:
        dz11=-dz11
    z14 = z0 + dz11

    point_3d.append([point[18] / s_target, point[19] / s_target, z14]) # 1
    dz9 = math.sqrt(max(L[8] ** 2 - ((point[14] - point[18]) ** 2 + (point[15] - point[19]) ** 2) / (s_target ** 2), 0))
    if point[31]<point[33]:
        dz9=-dz9
    z12 = z14 + dz9

    point_3d.append([point[14] / s_target, point[15] / s_target, z12]) # 2
    dz8 = math.sqrt(max(L[7] ** 2 - ((point[12] - point[14]) ** 2 + (point[13] - point[15]) ** 2) / (s_target ** 2), 0))
    if point[30]<point[31]:
        dz8=-dz8

    z11 = z12 + dz8
    point_3d.append([point[12] / s_target, point[13] / s_target, z11]) # 3
    dz10 = math.sqrt(
        max(L[9] ** 2 - ((point[12] - point[16]) ** 2 + (point[13] - point[17]) ** 2) / (s_target ** 2), 0))
    if point[32]<point[30]:
        dz10=-dz10
    z13 = z11 + dz10

    point_3d.append([point[16] / s_target, point[17] / s_target, z13]) # 4
    dz12 = math.sqrt(
        max(L[11] ** 2 - ((point[16] - point[20]) ** 2 + (point[17] - point[21]) ** 2) / (s_target ** 2), 0))
    if point[34]<point[32]:
        dz12=-dz12
    z15 = z13 + dz12

    point_3d.append([point[20] / s_target, point[21] / s_target, z15]) # 5
    dz6 = math.sqrt(max(L[5] ** 2 - ((point[2] - point[14]) ** 2 + (point[3] - point[15]) ** 2) / (s_target ** 2), 0))
    if point[25]<point[31]:
        dz6=-dz6
    z6 = z12 + dz6

    point_3d.append([point[2] / s_target, point[3] / s_target, z6])  # 6
    dz2 = math.sqrt(max(L[1] ** 2 - ((point[2] - point[6]) ** 2 + (point[3] - point[7]) ** 2) / (s_target ** 2), 0))
    if point[27]<point[25]:
        dz2=-dz2
    z8 = z6 + dz2

    point_3d.append([point[6] / s_target, point[7] / s_target, z8]) # 7
    dz4 = math.sqrt(max(L[3] ** 2 - ((point[6] - point[10]) ** 2 + (point[7] - point[11]) ** 2) / (s_target ** 2), 0))
    if point[29]<point[27]:
        dz4=-dz4
    z10 = z8 + dz4

    point_3d.append([point[10] / s_target, point[11] / s_target, z10]) # 8
    dz1 = math.sqrt(max(L[0] ** 2 - ((point[0] - point[2]) ** 2 + (point[1] - point[3]) ** 2) / (s_target ** 2), 0))
    if point[24]<point[25]:
        dz1=-dz1
    z5 = z6 + dz1

    point_3d.append([point[0] / s_target, point[1] / s_target, z5]) # 9
    dz3 = math.sqrt(max(L[2] ** 2 - ((point[0] - point[4]) ** 2 + (point[1] - point[5]) ** 2) / (s_target ** 2), 0))
    if point[26]<point[24]:
        dz3=-dz3
    z7 = z5 + dz3

    point_3d.append([point[4] / s_target, point[5] / s_target, z7]) #
    dz5 = math.sqrt(max(L[4] ** 2 - ((point[4] - point[8]) ** 2 + (point[5] - point[9]) ** 2) / (s_target ** 2), 0))
    if point[28]<point[26]:
        dz5=-dz5
    z9 = z7 + dz5
    point_3d.append([point[8] / s_target, point[9] / s_target, z9]) # 11

    return point_3d


# 单帧优化定义的目标函数
def f(s, point, s_target,L):
    dz1 = math.sqrt(max(L[0] ** 2 - ((point[0] - point[2]) ** 2 + (point[1] - point[3]) ** 2) / (s_target ** 2), 0))
    dz2 = math.sqrt(max(L[1] ** 2 - ((point[2] - point[6]) ** 2 + (point[3] - point[7]) ** 2) / (s_target ** 2), 0))
    dz3 = math.sqrt(max(L[2] ** 2 - ((point[0] - point[4]) ** 2 + (point[1] - point[5]) ** 2) / (s_target ** 2), 0))
    dz4 = math.sqrt(max(L[3] ** 2 - ((point[6] - point[10]) ** 2 + (point[7] - point[11]) ** 2) / (s_target ** 2), 0))
    dz5 = math.sqrt(max(L[4] ** 2 - ((point[4] - point[8]) ** 2 + (point[5] - point[9]) ** 2) / (s_target ** 2), 0))
    dz6 = math.sqrt(max(L[5] ** 2 - ((point[2] - point[14]) ** 2 + (point[3] - point[15]) ** 2) / (s_target ** 2), 0))
    dz8 = math.sqrt(max(L[7] ** 2 - ((point[12] - point[14]) ** 2 + (point[13] - point[15]) ** 2) / (s_target ** 2), 0))
    dz9 = math.sqrt(max(L[8] ** 2 - ((point[14] - point[18]) ** 2 + (point[15] - point[19]) ** 2) / (s_target ** 2), 0))
    dz10 = math.sqrt(
        max(L[9] ** 2 - ((point[12] - point[16]) ** 2 + (point[13] - point[17]) ** 2) / (s_target ** 2), 0))
    dz11 = math.sqrt(
        max(L[10] ** 2 - ((point[18] - point[22]) ** 2 + (point[19] - point[23]) ** 2) / (s_target ** 2), 0))
    dz12 = math.sqrt(
        max(L[11] ** 2 - ((point[16] - point[20]) ** 2 + (point[17] - point[21]) ** 2) / (s_target ** 2), 0))
    y = 0
    y += (s * math.sqrt(L[0] ** 2 - dz1 ** 2) - math.sqrt((point[0] - point[2]) ** 2 + (point[1] - point[3]) ** 2)) ** 2 +\
         (s * math.sqrt(L[1] ** 2 - dz2 ** 2) - math.sqrt((point[2] - point[6]) ** 2 + (point[3] - point[7]) ** 2)) ** 2 +\
         (s * math.sqrt(L[2] ** 2 - dz3 ** 2) - math.sqrt((point[0] - point[4]) ** 2 + (point[1] - point[5]) ** 2)) ** 2 +\
         (s * math.sqrt(L[3] ** 2 - dz4 ** 2) - math.sqrt((point[6] - point[10]) ** 2 + (point[7] - point[11]) ** 2)) ** 2 +\
         (s * math.sqrt(L[4] ** 2 - dz5 ** 2) - math.sqrt((point[4] - point[8]) ** 2 + (point[5] - point[9]) ** 2)) ** 2 +\
         (s * math.sqrt(L[5] ** 2 - dz6 ** 2) - math.sqrt((point[2] - point[14]) ** 2 + (point[3] - point[15]) ** 2)) ** 2 +\
         (s * math.sqrt(L[7] ** 2 - dz8 ** 2) - math.sqrt((point[12] - point[14]) ** 2 + (point[13] - point[15]) ** 2)) ** 2 +\
         (s * math.sqrt(L[8] ** 2 - dz9 ** 2) - math.sqrt((point[14] - point[18]) ** 2 + (point[15] - point[19]) ** 2)) ** 2 +\
         (s * math.sqrt(L[9] ** 2 - dz10 ** 2) - math.sqrt((point[12] - point[16]) ** 2 + (point[13] - point[17]) ** 2)) ** 2 +\
         (s * math.sqrt(L[10] ** 2 - dz11 ** 2) - math.sqrt((point[18] - point[22]) ** 2 + (point[19] - point[23]) ** 2)) ** 2 +\
         (s * math.sqrt(L[11] ** 2 - dz12 ** 2) - math.sqrt((point[16] - point[20]) ** 2 + (point[17] - point[21]) ** 2)) ** 2

    # print("dz!!!!!!!!!!!!!!!!!!!!!!!",dz1,dz2,dz3,dz4,dz5,dz6,dz8,dz9,dz10,dz11,dz12)
    # print("\n")


    return y

# 多帧优化定义的目标函数
def f_s(s, begin, end,worksheet1, L):
    z = 0
    for i in range(end - begin + 1):
        point = worksheet1.row_values(begin + i)
        point.remove(point[0])
        # s_target = get_s(point)
        z += f(s[i], point, s[i], L)
    return z

