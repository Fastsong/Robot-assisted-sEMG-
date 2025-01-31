import numpy as np
import pandas as pd
import datetime
from numpy import array
import scipy.io as scio
from datetime import datetime, time
import os

#读入肌电数据，并且调整通道
data1 = pd.read_csv('D:\sEMG_0909\SEMG\semg_9_16\myz\AG1.emt', skiprows=10, delimiter='\t', low_memory=False)
arr1 = data1.values
AGemg1 = np.copy(arr1)
AGemgg1 = AGemg1[:, 2:10]
AGemgg1[:, [2, 3, 4, 5, 6, 7, 0, 1]] = AGemgg1[:, [0, 1, 2, 3, 4, 5, 6, 7]]

data2 = pd.read_csv('D:\sEMG_0909\SEMG\semg_9_16\myz\AG2.emt', skiprows=10, delimiter='\t', low_memory=False)
arr2 = data2.values
AGemg2 = np.copy(arr2)
AGemgg2 = AGemg2[:, 2:10]
AGemgg2[:, [2, 3, 4, 5, 6, 7, 0, 1]] = AGemgg2[:, [0, 1, 2, 3, 4, 5, 6, 7]]

data3 = pd.read_csv('D:\sEMG_0909\SEMG\semg_9_16\myz\DX1.emt', skiprows=10, delimiter='\t', low_memory=False)
arr3 = data3.values
DXemg1 = np.copy(arr3)
DXemgg1 = DXemg1[:, 2:10]
DXemgg1[:, [2, 3, 4, 5, 6, 7, 0, 1]] = DXemgg1[:, [0, 1, 2, 3, 4, 5, 6, 7]]

data4 = pd.read_csv('D:\sEMG_0909\SEMG\semg_9_16\myz\DX2.emt', skiprows=10, delimiter='\t', low_memory=False)
arr4 = data4.values
DXemg2 = np.copy(arr4)
DXemgg2 = DXemg2[:, 2:10]
DXemgg2[:, [2, 3, 4, 5, 6, 7, 0, 1]] = DXemgg2[:, [0, 1, 2, 3, 4, 5, 6, 7]]

#读入坐标数据
reader1 = pd.read_csv('D:/sEMG_0909/AG_label/AG_9-16/Data-AG-12.csv', low_memory=False)
reader2 = pd.read_csv('D:/sEMG_0909/DX_label/DX_9-16/Data-DX-12.csv', low_memory=False)
ARR1 = reader1.values
ARR2 = reader2.values

reader3 = pd.read_csv('LIST1.csv',low_memory=False)



# 拿出AG主动位置坐标
zhukang1 = []
for i in range(0, len(ARR1)):
    L = ARR1[i]
    str1 = "2023-09-16 "
    lin = []
    if L[0] == '主动':
        lin.append(str1 + L[1][5:])
        lin.append('主动' + L[2][7:])
        zhukang1.append(lin)
        # index.append(i+1)

AGGc = np.array(zhukang1)

# 拿出AG被动坐标
beikang1 = []
for i in range(0,len(ARR1)):
    L = ARR1[i]
    str1 = "2023-09-16 "
    lin = []
    if L[0] == '被动':
        lin.append(str1 + L[1][5:])
        lin.append('被动' + L[2][7:])
        beikang1.append(lin)
        # index.append(i+1)

AGGd = np.array(beikang1)

# DX主动坐标提取
guancha21 = []
for i in range(0, len(ARR2)):
    L = ARR2[i]
    str1 = "2023-09-16 "
    lin = []
    if L[0] == '主动':
        lin.append(str1 + L[1])
        lin.append('主动' + L[4])
        guancha21.append(lin)

DXGc = np.array(guancha21)

# DX被动坐标提取
guancha22 = []
for i in range(0, len(ARR2)):
    L = ARR2[i]
    str1 = "2023-09-16 "
    lin = []
    if L[0] == '被动':
        lin.append(str1 + L[1])
        lin.append('被动' + L[4])
        guancha22.append(lin)

DXGd = np.array(guancha22)

startAG1 = reader3.iloc[11,1]   # (input('AG1时间段开始：'))
sAG1_k = datetime.strptime(startAG1, "%Y-%m-%d %H:%M:%S")
stopAG1 = AGGd[-1][0]
sAG1_s = datetime.strptime(stopAG1, "%Y-%m-%d %H:%M:%S:%f")

# AG1主动坐标提取切割

# 时间分割
AGtime1 = []
for i in range(0, len(AGGc)):
    L = datetime.strptime(AGGc[i][0], "%Y-%m-%d %H:%M:%S:%f")
    if sAG1_k <= L <= sAG1_s:
        AGtime1.append(AGGc[i])
AGtime1 = np.array(AGtime1)

# AG1被动时间提取切割
AGtime12 = []
for i in range(0, len(AGGd)):
    L = datetime.strptime(AGGd[i][0], "%Y-%m-%d %H:%M:%S:%f")
    if sAG1_k <= L <= sAG1_s:
        AGtime12.append(AGGd[i])
AGtime12 = np.array(AGtime12)


startAG2 = reader3.iloc[11,3]
# input('AG2时间段开始：'))
sAG2_k = datetime.strptime(startAG2, "%Y-%m-%d %H:%M:%S")
stopAG2 = AGGc[-1][0]
sAG2_s = datetime.strptime(stopAG2, "%Y-%m-%d %H:%M:%S:%f")

# AG2主动坐标
AGtime2 = []
for i in range(0, len(AGGc)):
    L = datetime.strptime(AGGc[i][0], "%Y-%m-%d %H:%M:%S:%f")
    if sAG2_k <= L <= sAG2_s:
        AGtime2.append(AGGc[i])
AGtime2 = np.array(AGtime2)


# DX1坐标确定
startDX1 = reader3.iloc[11,5]
sDX1_k = datetime.strptime(startDX1, "%Y-%m-%d %H:%M:%S")
stopDX1 = DXGd[-1][0]
sDX1_s = datetime.strptime(stopDX1, "%Y-%m-%d %H:%M:%S:%f")

# 分割主动动作
DXtime1 = []
for i in range(0, len(DXGc)):
    L = datetime.strptime(DXGc[i][0], "%Y-%m-%d %H:%M:%S:%f")
    if sDX1_k <= L <= sDX1_s:
        DXtime1.append(DXGc[i])
DXtime1 = np.array(DXtime1)

# 分割被动动作
DXtime12 = []
for i in range(0, len(DXGd)):
    L = datetime.strptime(DXGd[i][0], "%Y-%m-%d %H:%M:%S:%f")
    if sDX1_k <= L <= sDX1_s:
        DXtime12.append(DXGd[i])
DXtime12 = np.array(DXtime12)

startDX2 = reader3.iloc[11,7]
sDX2_k = datetime.strptime(startDX2, "%Y-%m-%d %H:%M:%S")
stopDX2 = DXGc[-1][0]
sDX2_s = datetime.strptime(stopDX2, "%Y-%m-%d %H:%M:%S:%f")

DXtime2 = []
for i in range(0, len(DXGc)):
    L = datetime.strptime(DXGc[i][0], "%Y-%m-%d %H:%M:%S:%f")
    if sDX2_k <= L <= sDX2_s:
        DXtime2.append(DXGc[i])
DXtime2 = np.array(DXtime2)



# check_time = datetime.strptime(check_time, '%H:%M:%S').time()
# start_time = datetime.strptime(start_time, '%H:%M:%S').time()
# end_time = datetime.strptime(end_time, '%H:%M:%S').time()
#
# # 判断check_time是否在start_time和end_time之间
# if start_time <= check_time <= end_time:
#     return True

# 目标点定位
AGduandian = ['x:-0.05743 y:0.649063', 'x:-0.124294 y:0.582199', 'x:-0.191158 y:0.515335',
              'x:0.0094341 y:0.621367', 'x:0.00943412 y:0.526807', 'x:0.0094341 y:0.432247',
              'x:0.0762983 y:0.649063', 'x:0.143162 y:0.582199', 'x:0.210027 y:0.515335']

# 利用动作点读肌电信号进行时间分割
# 主动动作肌电数据
AGzhengti1 = []
for i in range(0, len(AGtime1)):
    L = AGtime1[i][1]
    if L[2:] in AGduandian:
        if i < (len(AGtime1) - 2):
            linAG = []
            linAG1 = AGtime1[i:i + 3, :].tolist()
            linAG.append(linAG1)
            label = AGduandian.index(L[2:]) + 1
            linAG.append(str(label))
            AGzhengti1.append(linAG)

# 利用动作点读肌电信号进行时间分割
# 被动动作肌电数据
AGzhengti12 = []
for i in range(0, len(AGtime12)):
    L = AGtime12[i][1]
    if L[2:] in AGduandian:
        if i < (len(AGtime12) - 2):
            linAG = []
            linAG12 = AGtime12[i:i + 3, :].tolist()
            linAG.append(linAG12)
            label = AGduandian.index(L[2:]) + 1
            linAG.append(str(label))
            AGzhengti12.append(linAG)

AGzhengti2 = []
for i in range(0, len(AGtime2)):
    L = AGtime2[i][1]
    if L[2:] in AGduandian:
        if i < (len(AGtime2) - 2):
            linAG = []
            linAG1 = AGtime2[i:i + 3, :].tolist()
            linAG.append(linAG1)
            label = AGduandian.index(L[2:]) + 1
            linAG.append(str(label))
            AGzhengti2.append(linAG)



DXduandian = ['0.494015', '0.988031', '1.48353', '-0.494015', '-0.988031', '-1.48353']
# DXduandian = ['-0.371965', '-0.743929', '-1.11701', '0.371965', '0.743929', '1.11701']

DXzhengti1 = []
for i in range(0, len(DXtime1)):
    L = DXtime1[i][1]
    if L[18:] in DXduandian:
        if i < (len(DXtime1) - 2):
            linDX = []
            linDX.append(DXtime1[i:i+3, :])
            label = DXduandian.index(L[18:]) + 10
            linDX.append(label)
            DXzhengti1.append(linDX)

DXzhengti12 = []
for i in range(0, len(DXtime12)):
    L = DXtime12[i][1]
    if L[18:] in DXduandian:
        if i < (len(DXtime12) - 2):
            linDX = []
            linDX.append(DXtime12[i:i+3, :])
            label = DXduandian.index(L[18:]) + 10
            linDX.append(label)
            DXzhengti12.append(linDX)

DXzhengti2 = []
for i in range(0, len(DXtime2)):
    L = DXtime2[i][1]
    if L[18:] in DXduandian:
        if i < (len(DXtime2) - 2):
            linDX = []
            linDX.append(DXtime2[i:i+3, :])
            label = DXduandian.index(L[18:]) + 10
            linDX.append(label)
            DXzhengti2.append(linDX)


# df = pd.DataFrame(AGzhengti2)
# df.to_csv('D:/sEMG_0909/output2.csv', index=False, header=False)
#
# ce =input("截断，输入1")

# # 读取
# new_list = pd.read_csv('D:/sEMG_0909/output2.txt', header=None).values.tolist()
# import ast
# AGzhengti2= ast.literal_eval(new_list[0][0])

# 读取
# f = open('D:/sEMG_0909/output2.txt','r')
# x = f.readlines()
# def save_list_to_txt(lst, filename):
#     with open(filename, 'w') as f:
#         for item in lst:
#             f.write(str(item) + '\n')
#
# def read_list_from_txt(filename):
#     with open(filename, 'r') as f:
#         lines = f.readlines()
#         lst = [line.strip() for line in lines]
#     return lst
#
# filename = 'D:/sEMG_0909/output2.txt'
#
# save_list_to_txt(AGzhengti1, filename)
# new_lst1 = read_list_from_txt(filename)

#
# new_list = pd.read_csv('D:/sEMG_0909/output2.csv', header=None).values.tolist()
# import ast
# # for i range len(new_list):
# my_1 = ast.literal_eval(new_list[0][0])

# check_time = datetime.strptime(check_time, '%H:%M:%S').time()
# start_time = datetime.strptime(start_time, '%H:%M:%S').time()
# end_time = datetime.strptime(end_time, '%H:%M:%S').time()
#
# # 判断check_time是否在start_time和end_time之间
# if start_time <= check_time <= end_time:
#     return True


# df = pd.DataFrame(AGzhengti1)
# df.to_csv('D:/sEMG_0909/output1.txt', index=False, header=False)
#
# ce =input("截断，输入1")
#
# # 读取
# new_list = pd.read_csv('D:/sEMG_0909/output1.txt', header=None).values.tolist()
# import ast
# for i in range(0, len(new_list)):
#     new_list[i][0] = ast.literal_eval(new_list[i][0])
# AGzhengti1 = new_list
# # my_1 = ast.literal_eval(new_list[0][0])
#
#
# df2 = pd.DataFrame(AGzhengti2)
# df2.to_csv('D:/sEMG_0909/output2.txt', index=False, header=False)
#
# ce =input("截断，输入1")
#
# # 读取
# new_list2 = pd.read_csv('D:/sEMG_0909/output2.txt', header=None).values.tolist()
#
# import ast
# for i in range(0, len(new_list2)):
#     new_list2[i][0] = ast.literal_eval(new_list2[i][0])
# AGzhengti2 = new_list2



start1 = reader3.iloc[11,9]
starttimeAG1 = datetime.strptime(start1, "%Y-%m-%d %H:%M:%S")

start2 = reader3.iloc[11,10]
starttimeAG2 = datetime.strptime(start2, "%Y-%m-%d %H:%M:%S")

start3 = reader3.iloc[11,11]
starttimeDX1 = datetime.strptime(start3, "%Y-%m-%d %H:%M:%S")

start4 = reader3.iloc[11,12]
starttimeDX2 = datetime.strptime(start4, "%Y-%m-%d %H:%M:%S")

conditions = {1: ['Left', 'Distance1', 1], 2: ['Left', 'Distance2', 2], 3: ['Left', 'Distance3', 3],
              4: ['Vertical', 'Distance1', 4], 5: ['Vertical', 'Distance2', 5], 6: ['Vertical', 'Distance3', 6]
              , 7: ['Right', 'Distance1', 7], 8: ['Right', 'Distance2', 8], 9: ['Right', 'Distance3', 9]
              , 10: ['Direction1', 'Angle1', 10], 11: ['Direction1', 'Angle2', 11], 12: ['Direction1', 'Angle3', 12]
              , 13: ['Direction2', 'Angle1', 13], 14: ['Direction2', 'Angle2', 14], 15: ['Direction2', 'Angle3', 15]}


AG1_all = np.empty((len(AGzhengti1), 1), dtype=object)
AG2_all = np.empty((len(AGzhengti2), 1), dtype=object)
AG3_all = np.empty((len(AGzhengti12), 1), dtype=object)
DX1_all = np.empty((len(DXzhengti1), 1), dtype=object)
DX2_all = np.empty((len(DXzhengti2), 1), dtype=object)
DX3_all = np.empty((len(DXzhengti12), 1), dtype=object)

AG1_label = np.empty((len(AGzhengti1), 1),dtype=object)
AG2_label = np.empty((len(AGzhengti2), 1), dtype=object)
AG3_label = np.empty((len(AGzhengti12), 1), dtype=object)
DX1_label = np.empty((len(DXzhengti1), 1), dtype=object)
DX2_label = np.empty((len(DXzhengti2), 1), dtype=object)
DX3_label = np.empty((len(DXzhengti12), 1), dtype=object)

AG1_time = np.empty((len(AGzhengti1), 1), dtype=object)
AG2_time = np.empty((len(AGzhengti2), 1), dtype=object)
AG3_time = np.empty((len(AGzhengti12), 1), dtype=object)
DX1_time = np.empty((len(DXzhengti1), 1), dtype=object)
DX2_time = np.empty((len(DXzhengti2), 1), dtype=object)
DX3_time = np.empty((len(DXzhengti12), 1), dtype=object)

for i in range(0, len(AGzhengti1)):
    DDtime = AGzhengti1[i][0]
    time1 = datetime.strptime(DDtime[0][0], "%Y-%m-%d %H:%M:%S:%f")
    time2 = datetime.strptime(DDtime[2][0], "%Y-%m-%d %H:%M:%S:%f")
    # delta1 = time1 + datetime.timedelta(hours=8) - starttimeAG
    delta1 = time1 - starttimeAG1
    T1 = delta1.seconds * 1000 + delta1.microseconds/1000
    # delta2 = time2 + datetime.timedelta(hours=8) - starttimeAG
    delta2 = time2 - starttimeAG1
    T2 = delta2.seconds * 1000 + delta2.microseconds/1000
    motionlin = AGemgg1[int(T1):int(T2), :]
    AG1_all[i][0] = motionlin
    kong = []
    kong.append(AGzhengti1[i][0][0][0])
    kong.append(AGzhengti1[i][0][2][0])
    AG1_time[i][0] = kong
    motionlabel = int(AGzhengti1[i][1])
    linlabel = conditions[int(motionlabel)]
    AG1_label[i][0] = linlabel
# scio.savemat('D:/sEMG_0811/semg_quchu/fangan1/AG_CW_motion' + str(i) + '.mat', {'data': motionlin, 'label': motionlabel})

for i in range(0, len(AGzhengti12)):
    DDtime = AGzhengti12[i][0]
    time1 = datetime.strptime(DDtime[0][0], "%Y-%m-%d %H:%M:%S:%f")
    time2 = datetime.strptime(DDtime[2][0], "%Y-%m-%d %H:%M:%S:%f")
    # delta1 = time1 + datetime.timedelta(hours=8) - starttimeAG
    delta1 = time1 - starttimeAG1
    T1 = delta1.seconds * 1000 + delta1.microseconds/1000
    # delta2 = time2 + datetime.timedelta(hours=8)* - starttimeAG
    delta2 = time2 - starttimeAG1
    T2 = delta2.seconds * 1000 + delta2.microseconds/1000
    motionlin = AGemgg1[int(T1):int(T2), :]
    AG3_all[i][0] = motionlin
    kong = []
    kong.append(AGzhengti12[i][0][0][0])
    kong.append(AGzhengti12[i][0][2][0])
    AG3_time[i][0] = kong
    motionlabel = int(AGzhengti12[i][1])
    linlabel = conditions[int(motionlabel)]
    AG3_label[i][0] = linlabel

for i in range(0, len(AGzhengti2)):
    DDtime = AGzhengti2[i][0]
    time1 = datetime.strptime(DDtime[0][0], "%Y-%m-%d %H:%M:%S:%f")
    time2 = datetime.strptime(DDtime[2][0], "%Y-%m-%d %H:%M:%S:%f")
    # delta1 = time1 + datetime.timedelta(hours=8) - starttimeAG
    delta1 = time1 - starttimeAG2
    T1 = delta1.seconds * 1000 + delta1.microseconds/1000
    # delta2 = time2 + datetime.timedelta(hours=8) - starttimeAG
    delta2 = time2 - starttimeAG2
    T2 = delta2.seconds * 1000 + delta2.microseconds/1000
    motionlin = AGemgg2[int(T1):int(T2), :]
    AG2_all[i][0] = motionlin
    kong = []
    kong.append(AGzhengti2[i][0][0][0])
    kong.append(AGzhengti2[i][0][2][0])
    AG2_time[i][0] = kong
    motionlabel = int(AGzhengti2[i][1])
    linlabel = conditions[int(motionlabel)]
    AG2_label[i][0] = linlabel

for i in range(0, len(DXzhengti1)):
    DDtime = DXzhengti1[i][0]
    time1 = datetime.strptime(DDtime[0][0], "%Y-%m-%d %H:%M:%S:%f")
    time2 = datetime.strptime(DDtime[2][0], "%Y-%m-%d %H:%M:%S:%f")
    # delta1 = time1 + datetime.timedelta(hours=8) - starttimeAG
    delta1 = time1 - starttimeDX1
    T1 = delta1.seconds * 1000 + delta1.microseconds/1000
    # delta2 = time2 + datetime.timedelta(hours=8) - starttimeAG
    delta2 = time2 - starttimeDX1
    T2 = delta2.seconds * 1000 + delta2.microseconds/1000
    motionlin = DXemgg1[int(T1):int(T2), :]
    DX1_all[i][0] = motionlin
    kong = []
    kong.append(DXzhengti1[i][0][0][0])
    kong.append(DXzhengti1[i][0][2][0])
    DX1_time[i][0] = kong
    motionlabel = int(DXzhengti1[i][1])
    linlabel = conditions[int(motionlabel)]
    DX1_label[i][0] = linlabel

for i in range(0, len(DXzhengti12)):
    DDtime = DXzhengti12[i][0]
    time1 = datetime.strptime(DDtime[0][0], "%Y-%m-%d %H:%M:%S:%f")
    time2 = datetime.strptime(DDtime[2][0], "%Y-%m-%d %H:%M:%S:%f")
    # delta1 = time1 + datetime.timedelta(hours=8) - starttimeAG
    delta1 = time1 - starttimeDX1
    T1 = delta1.seconds * 1000 + delta1.microseconds/1000
    # delta2 = time2 + datetime.timedelta(hours=8) - starttimeAG
    delta2 = time2 - starttimeDX1
    T2 = delta2.seconds * 1000 + delta2.microseconds/1000
    motionlin = DXemgg1[int(T1):int(T2), :]
    DX3_all[i][0] = motionlin
    kong = []
    kong.append(DXzhengti12[i][0][0][0])
    kong.append(DXzhengti12[i][0][2][0])
    DX3_time[i][0] = kong
    motionlabel = int(DXzhengti12[i][1])
    linlabel = conditions[int(motionlabel)]
    DX3_label[i][0] = linlabel

for i in range(0, len(DXzhengti2)):
    DDtime = DXzhengti2[i][0]
    time1 = datetime.strptime(DDtime[0][0], "%Y-%m-%d %H:%M:%S:%f")
    time2 = datetime.strptime(DDtime[2][0], "%Y-%m-%d %H:%M:%S:%f")
    # delta1 = time1 + datetime.timedelta(hours=8) - starttimeAG
    delta1 = time1 - starttimeDX2
    T1 = delta1.seconds * 1000 + delta1.microseconds/1000
    # delta2 = time2 + datetime.timedelta(hours=8) - starttimeAG
    delta2 = time2 - starttimeDX2
    T2 = delta2.seconds * 1000 + delta2.microseconds/1000
    motionlin = DXemgg2[int(T1):int(T2), :]
    DX2_all[i][0] = motionlin
    kong = []
    kong.append(DXzhengti2[i][0][0][0])
    kong.append(DXzhengti2[i][0][2][0])
    DX2_time[i][0] = kong
    motionlabel = int(DXzhengti2[i][1])
    linlabel = conditions[int(motionlabel)]
    DX2_label[i][0] = linlabel

import matplotlib.pyplot as plt
x = np.arange(len(AG1_all[1][0][:]))
plt.figure()# 创建一个灰色画布
plt.plot(x,AG1_all[1][0][:])              # 绘制data数据的折线图
plt.show()

AGZ_all = np.concatenate((AG1_all, AG2_all))
AGZ_time = np.concatenate((AG1_time,AG2_time))
AGZ_label = np.concatenate((AG1_label,AG2_label))
DXZ_all = np.concatenate((DX1_all, DX2_all))
DXZ_time = np.concatenate((DX1_time,DX2_time))
DXZ_label = np.concatenate((DX1_label,DX2_label))

# print(AGZ_all[0][0][:])
import matplotlib.pyplot as plt
x = np.arange(len(AGZ_all[1][0][:]))
plt.figure()# 创建一个灰色画布
plt.plot(x,AGZ_all[1][0][:])              # 绘制data数据的折线图
plt.show()

# print(DXZ_all[0][0][:])
import matplotlib.pyplot as plt
x = np.arange(len(DXZ_all[1][0][:]))
plt.figure()# 创建一个灰色画布
plt.plot(x,DXZ_all[1][0][:])              # 绘制data数据的折线图
plt.show()

scio.savemat('D:/sEMG_0909/Split/S12_data1.mat',
             {'subset1_data': AGZ_all, 'subset1_time': AGZ_time, 'subset1_label': AGZ_label
              , 'subset2_data': AG3_all, 'subset2_time': AG3_time, 'subset2_label': AG3_label})
scio.savemat('D:/sEMG_0909/Split/S12_data2.mat',
             {'subset1_data': DXZ_all, 'subset1_time': DXZ_time, 'subset1_label': DXZ_label
              , 'subset2_data': DX3_all, 'subset2_time': DX3_time, 'subset2_label': DX3_label})

stop = 0

