import numpy as np
import pandas as pd
import datetime
from numpy import array
import scipy.io as scio
from datetime import datetime, time


#读入坐标数据
reader1 = pd.read_csv(r'D:\SEMG\sEMGdata\fmj\AG-0919-FANG.csv', low_memory=False)
reader2 = pd.read_csv(r'D:\SEMG\sEMGdata\fmj\DX-0919-FANG.csv', low_memory=False)
ARR1 = reader1.values
ARR2 = reader2.values

#reader3 = pd.read_excel('D:\SEMG\sEMGdata\time.xlsx',sheet_name='Sheet1',header=0)
reader3 = pd.read_excel(r'D:\SEMG\sEMGdata\time.xlsx')

str1 = "2024-09-10 "

# 拿出AG主动位置坐标
zhukang1 = []
idx1 = []
for i in range(0, len(ARR1)):
    L = ARR1[i]
    lin = []
    idx=[]
    velocity=[]
    if L[0] == 'mode:抗阻':
        if ARR1[i][2] != ARR1[i-1][2]:
            lin.append(str1 + L[1][5:])
            lin.append('主动' + L[2][7:])
            lin.append(i)
            zhukang1.append(lin)
            # index.append(i+1)

AGGc = np.array(zhukang1)


# 拿出AG被动坐标
beikang1 = []
for i in range(0,len(ARR1)):
    L = ARR1[i]
    lin = []
    if L[0] == 'mode:被动' :
        if ARR1[i][2] != ARR1[i-1][2]:
            lin.append(str1 + L[1][5:])
            lin.append('被动' + L[2][7:])
            lin.append(i)
            beikang1.append(lin)
            # index.append(i+1)

AGGd = np.array(beikang1)


# 导出数据分割的信息，准备删除标签


# DX主动坐标提取
guancha21 = []
for i in range(0, len(ARR2)):
    L = ARR2[i]
    lin = []
    if L[0] == '主动':        
        lin.append(str1 + L[1])
        lin.append('主动' + L[4])
        lin.append(i)
        guancha21.append(lin)

DXGc = np.array(guancha21)

# DX被动坐标提取
guancha22 = []
for i in range(0, len(ARR2)):
    L = ARR2[i]
    lin = []
    if L[0] == '被动':
        lin.append(str1 + L[1])
        lin.append('被动' + L[4])
        lin.append(i)
        guancha22.append(lin)

DXGd = np.array(guancha22)

# 时间确定
startAG1 = str(reader3.iloc[15,1])   # (input('AG1时间段开始：'))
startAG1 = str1 + startAG1
sAG1_k = datetime.strptime(startAG1, "%Y-%m-%d %H:%M:%S")
stopAG1 = str(reader3.iloc[15,2])
stopAG1 = str1 + stopAG1
sAG1_s = datetime.strptime(stopAG1, "%Y-%m-%d %H:%M:%S")



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





# DX1坐标确定
startDX1 = str(reader3.iloc[15,3])
startDX1 = str1 + startDX1
sDX1_k = datetime.strptime(startDX1, "%Y-%m-%d %H:%M:%S")
stopDX1 = str(reader3.iloc[15,4]) 
stopDX1 = str1 + stopDX1
sDX1_s = datetime.strptime(stopDX1, "%Y-%m-%d %H:%M:%S")


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





# 目标点定位
AGduandian = [
                'x:-0.00575741 y:0.682677','x:0.0725923 y:0.604327', 'x:0.150942 y:0.525977', 
                'x:-0.0841072 y:0.650223' ,'x:-0.0841072 y:0.53942', 'x:-0.0841072 y:0.428617',
                'x:-0.162457 y:0.682677'  ,'x:-0.240807 y:0.604327', 'x:-0.319156 y:0.525977',    
              ]
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




# DXduandian = ['0.494015', '0.988031', '1.48353', '-0.494015', '-0.988031', '-1.48353']
DXduandian = ['-0.371965', '-0.743929', '-1.11701', '0.371965', '0.743929', '1.11701']
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



CutNumberAG1 = []
for i in range(0,len(AGzhengti1)):
    cut = []
    cut.append(AGzhengti1[i][0][0][2])
    cut.append(AGzhengti1[i][0][1][2])
    CutNumberAG1.append(cut)

CutNumberAG12 = []
for i in range(0,len(AGzhengti12)):
    cut = []
    cut.append(AGzhengti12[i][0][0][2])
    cut.append(AGzhengti12[i][0][1][2])
    CutNumberAG12.append(cut)


CutNumberDX1 = []
for i in range(0,len(DXzhengti1)):
    cut = []
    cut.append(DXzhengti1[i][0][0][2])
    cut.append(DXzhengti1[i][0][1][2])
    CutNumberDX1.append(cut)

CutNumberDX12 = []
for i in range(0,len(DXzhengti12)):
    cut = []
    cut.append(DXzhengti12[i][0][0][2])
    cut.append(DXzhengti12[i][0][1][2])
    CutNumberDX12.append(cut)


# AG1主动坐标提取切割
AGP1_all = np.empty((len(AGzhengti1), 1), dtype=object)
AGP2_all = np.empty((len(AGzhengti12), 1), dtype=object)

AGV1_all = np.empty((len(AGzhengti1), 1), dtype=object)
AGV2_all = np.empty((len(AGzhengti12), 1), dtype=object)

AGT1_all = np.empty((len(AGzhengti1), 1), dtype=object)
AGT2_all = np.empty((len(AGzhengti12), 1), dtype=object)

DXP1_all = np.empty((len(DXzhengti1), 1), dtype=object)
DXP2_all = np.empty((len(DXzhengti12), 1), dtype=object)

DXV1_all = np.empty((len(DXzhengti1), 1), dtype=object)
DXV2_all = np.empty((len(DXzhengti12), 1), dtype=object)

DXM1_all = np.empty((len(DXzhengti1), 1), dtype=object)
DXM2_all = np.empty((len(DXzhengti12), 1), dtype=object)

DXS1_all = np.empty((len(DXzhengti1), 1), dtype=object)
DXS2_all = np.empty((len(DXzhengti12), 1), dtype=object)

# AgVelocity1 = []
# AgTorq1 = []
for i in range(0,len(CutNumberAG1)):
    T1=CutNumberAG1[i][0]
    T2=CutNumberAG1[i][1]
    AP = ARR1[(int(T1) + 1):int(T2), 1]
    AGP1_all[i][0] = AP
    AV=ARR1[(int(T1)+1):int(T2),2]
    AGV1_all[i][0] = AV
    AT = ARR1[(int(T1)+1):int(T2),3]
    AGT1_all[i][0] = AT
    # AgVelocity1.append(AV)
    # AgTorq1.append(AT)

# AgVelocity12 = []
# AgTorq12 = []
for i in range(0,len(CutNumberAG12)):
    T1=CutNumberAG12[i][0]
    T2=CutNumberAG12[i][1]
    AP = ARR1[(int(T1) + 1):int(T2), 1]
    AGP2_all[i][0] = AP
    AV=ARR1[(int(T1)+1):int(T2),2]
    AGV2_all[i][0] = AV
    AT = ARR1[(int(T1)+1):int(T2),3]
    AGT2_all[i][0] = AT






# DxVelocity1 = []
# DxMotor1 = []
# DxSensor1 = []
for i in range(0,len(CutNumberDX1)):
    T1=CutNumberDX1[i][0]
    T2=CutNumberDX1[i][1]
    DP=ARR2[(int(T1) + 1):int(T2), 1]
    DV=ARR2[(int(T1)+1):int(T2),2]
    DM=ARR2[(int(T1)+1):int(T2),3]
    DS=ARR2[(int(T1)+1):int(T2),4]
    DXP1_all[i][0] = DP
    DXV1_all[i][0] = DV
    DXM1_all[i][0] = DM
    DXS1_all[i][0] = DS


    # DxVelocity1.append(DV)
    # DxMotor1.append(DM)
    # DxSensor1.append(DS)




# DxVelocity12 = []
# DxMotor12 = []
# DxSensor12 = []
for i in range(0,len(CutNumberDX12)):
    T1=CutNumberDX12[i][0]
    T2=CutNumberDX12[i][1]
    DP = ARR2[(int(T1) + 1):int(T2), 1]
    DV=ARR2[(int(T1)+1):int(T2),2]
    DM=ARR2[(int(T1)+1):int(T2),3]
    DS=ARR2[(int(T1)+1):int(T2),4]
    DXP2_all[i][0] = DP
    DXV2_all[i][0] = DV
    DXM2_all[i][0] = DM
    DXS2_all[i][0] = DS
    # DxVelocity12.append(DV)
    # DxMotor12.append(DM)
    # DxSensor12.append(DS)

# for i in range(0, len(ARR2)):
#     L = ARR2[i]
#     str1 = "2023-09-09 "
#     velocity=[]
#     Mtorq=[]
#     Storq=[]
#     if L[3] == np.nan:
#         continue
#     velocity.append(str1 + L[0][5:])
#     velocity.append(L[2])
#     Mtorq.append(str1 + L[0][5:])
#     Mtorq.append(L[3])
#     Storq.append(str1 + L[0][5:])
#     Storq.append(L[4])
#     DxVelocity.append(velocity)
#     DxMotor.append(Mtorq)
#     DxSensor.append(Storq)
#
# DxVc = np.array(DxVelocity)
# DxMc = np.array(DxMotor)
# DxSc = np.array(DxSensor)

# AgVelocity = []
# AgTorq = []
# for i in range(0, len(ARR1)):
#     L = ARR1[i]
#     str1 = "2023-09-09 "
#     velocity=[]
#     torq=[]
#     if L[3] is np.nan:
#         continue
#     velocity.append(str1 + L[0][5:])
#     velocity.append(L[2])
#     torq.append(str1 + L[0][5:])
#     torq.append(L[3])
#     AgVelocity.append(velocity)
#     AgTorq.append(torq)
# AgVc = np.array(AgVelocity)
# AgTc = np.array(AgTorq)

# def find_deleted_lines(file_a, file_b):
#     with open(file_a, 'r') as f_a, open(file_b, 'r') as f_b:
#         lines_a = f_a.readlines()
#         lines_b = f_b.readlines()
#
#     deleted_lines = []
#     for i, line in enumerate(lines_a):
#         if line not in lines_b:
#             deleted_lines.append(i + 1)  # 行号从1开始计数
#
#     return deleted_lines
#
# # 示例用法
# file_a = 'file_a.txt'
# file_b = 'file_b.txt'
# deleted_lines = find_deleted_lines(file_a, file_b)
#
# if deleted_lines:
#     print("已删除的行号：")
#     for line_number in deleted_lines:
#         print(line_number)
# else:
#     print("没有删除的行。")




# AGZ_all = np.concatenate((AG1_all, AG2_all))
# AGZ_time = np.concatenate((AG1_time,AG2_time))
# AGZ_label = np.concatenate((AG1_label,AG2_label))
# DXZ_all = np.concatenate((DX1_all, DX2_all))
# DXZ_time = np.concatenate((DX1_time,DX2_time))
# DXZ_label = np.concatenate((DX1_label,DX2_label))

# DXZ_time1=[]
# for i in range(0,len(DXZ_time)):
#     DXZ_time1.append(DXZ_time[i][0])
#
# AGZ_time1=[]
# for i in range(0,len(AGZ_time)):
#     AGZ_time1.append(AGZ_time[i][0])

# AgZV = []
# for idx,time in enumerate(AGZ_time1):
#     ST1 = datetime.strptime(time[0], "%Y-%m-%d %H:%M:%S:%f")
#     SP2 = datetime.strptime(time[1], "%Y-%m-%d %H:%M:%S:%f")
#     time_start = 0
#     time_end = 0
#     for j,time_V in enumerate(AgVelocity):
#         L = datetime.strptime(time_V[0], "%Y-%m-%d %H:%M:%S:%f")
#         if L<=ST1:
#             time_start = j
#
#         continue
#
#         if L<=SP2:
#                 time_end = j
#                 break
#         # kong=[int(time_start):int(time_end),:]
#         AgZV.append(kong)
#
# AgVelocity = np.array(AgZV)

# def findlabel(x,y,z):


# for i in range(0, len(AGzhengti1)):
#     L = datetime.strptime(DXGd[i][0], "%Y-%m-%d %H:%M:%S:%f")
#
#     if sDX1_k <= L <= sDX1_s:
#         DXtime12.append(DXGd[i])
# DXtime12 = np.array(DXtime12)



data1 = scio.loadmat(r'D:\SEMG\sEMGdata\data\split\S5-3_data1.mat')
data2 = scio.loadmat(r'D:\SEMG\sEMGdata\data\split\S5-3_data2.mat')
AgZT = data1['subset1_time']
AgBT = data1['subset2_time']
AgZL = data1['subset1_label']
AgBL = data1['subset2_label']

DxZT = data2['subset1_time']
DxBT = data2['subset2_time']
DxZL = data2['subset1_label']
DxBL = data2['subset2_label']


scio.savemat('D:\SEMG\sEMGdata\data\split\S5-3_data3.mat',
             {'subset1_position': AGP1_all, 'subset1_velocity':AGV1_all, 'subset1_torq':AGT1_all, 'subset1_time': AgZT, 'subset1_label': AgZL
              , 'subset2_position': AGP2_all, 'subset2_velocity':AGV2_all, 'subset2_torq':AGT2_all, 'subset2_time': AgBT, 'subset2_label': AgBL})
scio.savemat('D:\SEMG\sEMGdata\data\split\S5-3_data4.mat',
             {'subset1_position': DXP1_all, 'subset1_velocity':DXV1_all, 'subset1_motor_torq':DXM1_all,'subset1_sensed_torq':DXS1_all,'subset1_time': DxZT, 'subset1_label': DxZL
              , 'subset2_position': DXP2_all, 'subset2_velocity':DXV2_all, 'subset2_motor_torq':DXM2_all,'subset2_sensed_torq':DXS2_all, 'subset2_time': DxBT, 'subset2_label': DxBL})
stop = 0

