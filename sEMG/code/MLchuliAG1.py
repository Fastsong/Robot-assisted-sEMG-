import numpy as np
import pandas as pd
import datetime
from numpy import array
import scipy.io as scio
from scipy import signal
import matplotlib.pyplot as plt
from numpy import array
import pywt
from scipy.signal import filtfilt, iirnotch, freqz, butter
from scipy.fftpack import fft, fftshift, fftfreq
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import math

def calculate_rms(data):
    """
    计算给定数据的RMS
    参数：
    data: 包含数据的列表或数组
    返回值：
    RMS
    """
    n = len(data)
    square_sum = sum(x**2 for x in data)
    rms = math.sqrt(square_sum / n)
    return rms


def calculate_mav(data):
    """
    计算给定数据的MAV
    参数：
    data: 包含数据的列表或数组
    返回值：
    MAV
    """
    n = len(data)
    mav = sum(abs(x) for x in data) / n
    return mav


def calculate_wl(data):
    """
    计算给定数据的WL
    参数：
    data: 包含数据的列表或数组
    返回值：
    WL
    """
    n = len(data)
    wl = sum(abs(data[i] - data[i-1]) for i in range(1, n))
    return wl


def calculate_zc(data):
    """
    计算给定数据的ZC
    参数：
    data: 包含数据的列表或数组
    返回值：
    ZC
    """
    n = len(data)
    zc = 0
    for i in range(1, n):
        if data[i] * data[i-1] < 0:
            zc += 1
    return zc


def calculate_ssc(data):
    """
    计算给定数据的SSC
    参数：
    data: 包含数据的列表或数组
    返回值：
    SSC
    """
    n = len(data)
    ssc = 0
    for i in range(1, n-1):
        if (data[i] > data[i-1] and data[i] > data[i+1]) or \
           (data[i] < data[i-1] and data[i] < data[i+1]):
            ssc += 1
    return ssc


def median_frequency(signal, sample_rate):
    # 对信号进行傅里叶变换
    n = len(signal)
    freqs = fftfreq(n, d=1 / sample_rate)
    complex_spectrum = fft(signal)
    power_spectrum = np.abs(complex_spectrum) ** 2

    # 计算频率轴上的中位数
    sorted_power_spectrum = sorted(power_spectrum)
    cumsum = np.cumsum(sorted_power_spectrum)
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    median_freq = freqs[power_spectrum.tolist().index(sorted_power_spectrum[median_idx])]

    return median_freq


def power_spectral_entropy(x, fs):
    # x: 输入信号
    # fs: 采样频率
    N = len(x)
    # 加窗和零填充
    x_p = x * signal.windows.hann(N)
    x_p = np.append(x_p, np.zeros(N))
    # 计算 PSD
    P = np.abs(np.fft.fft(x_p))**2 / (fs*N)
    P = P[:int(N/2)+1]
    P = P / np.sum(P)
    # 计算功率谱熵
    H = -np.sum(P*np.log2(P))
    return H

def AverageAmptitudeChange(X):
    N = len(X)
    Y = 0
    for i in range(0, N-1):
        Y = Y + abs(X[i+1] - X[i])
    AAC = Y/N
    return AAC

def RootMeanSquare(X):
    RMS = np.sqrt(np.mean(X*X))
    return RMS

def ModifiedMeanAbsoluteValue(X):
    MAV = np.mean(abs(X))
    return MAV


scaler = MinMaxScaler(feature_range=(-1, 1))



data_dict = {}


# # 循环处理每个文件
# for i in range(len(numbers)):
#     file_name = f'D:\SEMG\sEMGdata\data\split\AG1\{numbers[i]}.mat'
#     mat_data = scio.loadmat(file_name)

#     # 从MAT文件中提取数据并存储到字典中
#     data_data1= mat_data['subset1_data']
#     data_data2= mat_data['subset2_data']
#     data_data= np.vstack((data_data1,data_data2))
#     data_label1= mat_data['subset1_label']
#     data_label2= mat_data['subset2_label']
#     data_label= np.vstack((data_label1,data_label2))
#     data_dict[f'Data{i}'] = data_data
#     data_dict[f'Label{i}'] = data_label
# 循环处理每个文件
for i in range(16):
    if i == 15:
        file_name = f'D:\SEMG\sEMGdata\data\split\AG1\{i+1}.mat'
        mat_data = scio.loadmat(file_name)

        # 从MAT文件中提取数据并存储到字典中
        
        
        data_data= mat_data['subset2_data']
        data_label= mat_data['subset2_label']
        data_dict[f'Data{i}'] = data_data
        data_dict[f'Label{i}'] = data_label
    else:
        file_name = f'D:\SEMG\sEMGdata\data\split\AG1\{i+1}.mat'
        mat_data = scio.loadmat(file_name)

        # 从MAT文件中提取数据并存储到字典中
        data_data1= mat_data['subset1_data']
        data_data2= mat_data['subset2_data']
        data_data= np.vstack((data_data1,data_data2))
        data_label1= mat_data['subset1_label']
        data_label2= mat_data['subset2_label']
        data_label= np.vstack((data_label1,data_label2))
        data_dict[f'Data{i}'] = data_data
        data_dict[f'Label{i}'] = data_label



label9fenlei_dict = {}


for i in range(16):
    str1 = 'Label' + str(i)
    labellin1 = data_dict[str1]
    labelnew = []
    for j in range(len(labellin1)):
        labelraw = int(labellin1[j][0][2]) - 1
        # labellinlin = assign_label_147_fangxiang(labelraw)
        labelnew.append(labelraw)
    labelnew = np.array(labelnew)
    label9fenlei_dict[f'Label{i}'] = labelnew


def adjust_timeseries_length_v2(timeseries, target_length=5000):
    """
    调整8通道时间序列的长度，使其达到目标长度。
    此版本处理的时间序列格式为 (L, 8)，并在头尾进行补零。

    参数:
    timeseries (numpy.ndarray): 一个形状为 (L, 8) 的数组，其中 L 可以对于每个时间序列有所不同。
    target_length (int): 时间序列应调整到的目标长度。

    返回:
    numpy.ndarray: 调整后的时间序列，其形状为 (target_length, 8)。
    """
    current_length = timeseries.shape[0]

    if current_length < target_length:
        # 计算需要补零的总长度
        padding_length = target_length - current_length
        # 将补零平均分配到头部和尾部
        padding_before = padding_length // 2
        padding_after = padding_length - padding_before
        padding_before = np.zeros((padding_before, 8))
        padding_after = np.zeros((padding_after, 8))
        # 将补零添加到时间序列的头部和尾部
        adjusted_timeseries = np.vstack((padding_before, timeseries, padding_after))

    elif current_length > target_length:
        # # 如果时间序列太长，则从中间截断
        # trim_length = (current_length - target_length) // 2
        # adjusted_timeseries = timeseries[trim_length:trim_length + target_length]
        adjusted_timeseries = signal.resample(timeseries, target_length, axis=0)

    else:
        adjusted_timeseries = timeseries

    return adjusted_timeseries


def adjust_timeseries_length_v3(timeseries, target_length=5000):
    """
    调整时间序列的长度，并删除特定的列（第 3 和第 4 列）。
    此版本处理的时间序列格式为 (L, 8)，并在头尾进行补零。
    结果时间序列的形状为 (target_length, 6)。

    参数:
    timeseries (numpy.ndarray): 形状为 (L, 8) 的数组，L 可以对于每个时间序列有所不同。
    target_length (int): 时间序列应调整到的目标长度。

    返回:
    numpy.ndarray: 调整后的时间序列，形状为 (target_length, 6)。
    """
    current_length = timeseries.shape[0]

    if current_length < target_length:
        # 计算需要补零的总长度
        padding_length = target_length - current_length
        # 将补零平均分配到头部和尾部
        padding_before = padding_length // 2
        padding_after = padding_length - padding_before
        padding_before = np.zeros((padding_before, 8))
        padding_after = np.zeros((padding_after, 8))
        # 将补零添加到时间序列的头部和尾部
        adjusted_timeseries = np.vstack((padding_before, timeseries, padding_after))

    elif current_length > target_length:
        # 使用信号处理进行重新采样
        adjusted_timeseries = signal.resample(timeseries, target_length, axis=0)

    else:
        adjusted_timeseries = timeseries

    # 删除第 3 和第 4 列
    adjusted_timeseries = np.delete(adjusted_timeseries, [2, 3], axis=1)

    return adjusted_timeseries



for i in range(16):
    data_name = 'Data' + str(i)  # 构建变量名字符串
    data_lin = data_dict[data_name]

    processed_data_all = []

    for j in range(0, len(data_lin)):
        datalin0 = data_lin[j][0]

        preprocessed_data = []
        new_length = None

        for channel in range(8):
            channel_data = datalin0[:, channel]

            sos = signal.butter(10, 499, btype='low', output='sos', fs=1000)

            LF_x1 = signal.sosfilt(sos, channel_data)

            fs = 1000
            f0 = 50
            w0 = f0 / (fs / 2)
            Q = 30
            b, a = iirnotch(w0, Q)
            # filter response
            w, h = freqz(b, a)
            filt_freq = w * fs / (2 * np.pi)
            y_50Hz = filtfilt(b, a, LF_x1)

            threshold = 0.1
            w = pywt.Wavelet('db8')
            maxlev = pywt.dwt_max_level(len(y_50Hz), w.dec_len)
            coffs = pywt.wavedec(y_50Hz, 'db8', level=maxlev)
            for n in range(1, len(coffs)):
                coffs[n] = pywt.threshold(coffs[n], threshold * max(abs(coffs[n])))

            xiaobo = pywt.waverec(coffs, 'db8')
            processed_channel_data = xiaobo[1:]

            cyy_2d = processed_channel_data.reshape(-1, 1)
            data_normalized = scaler.fit_transform(cyy_2d)
            data_normalized1 = data_normalized.reshape(-1)

            preprocessed_data.append(data_normalized1)

            if new_length is None:
                new_length = len(processed_channel_data)
            else:
                new_length = min(new_length, len(processed_channel_data))

        preprocessed_data = [channel[:new_length] for channel in preprocessed_data]

        # 将每个通道的数据合并为一个二维数组
        preprocessed_data = np.column_stack(preprocessed_data)

        processed_data_all.append(preprocessed_data)

    # new_var_name = "Datanew" + str(i)  # 构建新变量名字符串
    processed_data_all = np.array(processed_data_all, dtype=object)
    # globals()[new_var_name] = processed_data_all
    label_var_name = 'Label' + str(i)
    label_lin1 = label9fenlei_dict[label_var_name] + 1
    # label_lin1 = 0

    AGdatalin = np.zeros((processed_data_all.shape[0], 6 * 8 * 5))

    for num in range(processed_data_all.shape[0]):
        datalin0 = processed_data_all[num]

        L = len(datalin0) // 6  # 窗长的大小
        d = len(datalin0) % 6  # 需要减去的余数
        datalin0 = datalin0[d:, :]
        RMSj = np.zeros((6, 8))
        WLj = np.zeros((6, 8))
        MAVj = np.zeros((6, 8))
        SSCj = np.zeros((6, 8))
        # zcj = np.zeros((6, 8))
        # mfj = np.zeros((6, 8))
        powj = np.zeros((6, 8))
        # AACj = np.zeros((6, 8))
        # RMSj = np.zeros((6, 8))
        # MAVj = np.zeros((6, 8))
        for m in range(0, 8):
            linP = datalin0[:, m]
            for n in range(0, 6):
                if n != 5:
                    jisuan = linP[0 + L * n: L + L * (n + 1) + 1]
                    rms = calculate_rms(jisuan)
                    wl = calculate_wl(jisuan)
                    mav = calculate_mav(jisuan)
                    ssc = calculate_ssc(jisuan)
                    # mf = median_frequency(jisuan, 1000)
                    pow = power_spectral_entropy(jisuan, 1000)
                    # zc = calculate_zc(jisuan)
                    RMSj[n, m] = rms
                    WLj[n, m] = wl
                    MAVj[n, m] = mav
                    SSCj[n, m] = ssc
                    # ZCj[n, m] = zc
                    # mfj[n, m] = mf
                    powj[n, m] = pow

                    # AAC = AverageAmptitudeChange(jisuan)
                    # AACj[n, m] = AAC
                    # RMS = RootMeanSquare(jisuan)
                    # RMSj[n, m] = RMS
                    # MAV = ModifiedMeanAbsoluteValue(jisuan)
                    # MAVj[n, m] = MAV
                else:
                    jisuan = linP[L * 5 + 1:]
                    rms = calculate_rms(jisuan)
                    wl = calculate_wl(jisuan)
                    mav = calculate_mav(jisuan)
                    ssc = calculate_ssc(jisuan)
                    # mf = median_frequency(jisuan, 1000)
                    pow = power_spectral_entropy(jisuan, 1000)
                    # zc = calculate_zc(jisuan)
                    RMSj[n, m] = rms
                    WLj[n, m] = wl
                    MAVj[n, m] = mav
                    SSCj[n, m] = ssc
                    # ZCj[n, m] = zc
                    # mfj[n, m] = mf
                    powj[n, m] = pow
                    # AAC = AverageAmptitudeChange(jisuan)
                    # AACj[n, m] = AAC
                    # RMS = RootMeanSquare(jisuan)
                    # RMSj[n, m] = RMS
                    # MAV = ModifiedMeanAbsoluteValue(jisuan)
                    # MAVj[n, m] = MAV
        feature1 = RMSj.flatten('F')
        feature2 = WLj.flatten('F')
        feature3 = MAVj.flatten('F')
        feature4 = SSCj.flatten('F')
        # feature5 = mfj.flatten('F')
        feature5 = powj.flatten('F')
        featurezong = np.append(feature1, feature2)
        featurezong = np.append(featurezong, feature3)
        featurezong = np.append(featurezong, feature4)
        featurezong = np.append(featurezong, feature5)
        # featurezong = np.append(featurezong, feature6)
        AGdatalin[num, :] = featurezong

    new_chuli_data = "Datachuli" + str(i)
    globals()[new_chuli_data] = AGdatalin
    label_chuli = 'Labelchuli' + str(i)
    globals()[label_chuli] = label_lin1
    scio.savemat('D:\SEMG\data\prodata\ML1\AG1_' + str(i)+'.mat',
                 {'data': AGdatalin, 'label': label_lin1})




    # Data_chuli = []
    # for m in range(len(processed_data_all)):
    #     lindata = processed_data_all[m]
    #     chudata = adjust_timeseries_length_v2(lindata, target_length=4000)
    #     chudata = chudata.T
    #     labelt = label_lin1[m]
    #     lin = [chudata, labelt]
    #     Data_chuli.append(lin)
    # Data_chuli = np.array(Data_chuli)

    # scio.savemat('D:/sEMG_1204/processeddata/adjust2/Num' + str(i) + '_data.mat', {'data': Data_chuli})

    # scio.savemat('D:/sEMG_1204/prodata/Num' + str(i) + '_data.mat',
    #              {'data': processed_data_all, 'label': label_lin1})


#
# for i in range(1, 5):
#     data_name = "Data" + str(i)  # 构建变量名字符串
#     data_lin = globals()[data_name]
#     label_name = "Label" + str(i)
#     label_lin = globals()[label_name]
#
#     processed_data_all = []
#
#     for j in range(0, len(data_lin)):
#         datalin0 = data_lin[j][0]
#
#         preprocessed_data = []
#         new_length = None
#
#         for channel in range(8):
#             channel_data = datalin0[:, channel]
#
#             sos = signal.butter(10, 499, btype='low', output='sos', fs=1000)
#
#             LF_x1 = signal.sosfilt(sos, channel_data)
#
#             fs = 1000
#             f0 = 50
#             w0 = f0 / (fs / 2)
#             Q = 30 
#             b, a = iirnotch(w0, Q)
#             # filter response
#             w, h = freqz(b, a)
#             filt_freq = w * fs / (2 * np.pi)
#             y_50Hz = filtfilt(b, a, LF_x1)
#
#             threshold = 0.1
#             w = pywt.Wavelet('db8')
#             maxlev = pywt.dwt_max_level(len(y_50Hz), w.dec_len)
#             coffs = pywt.wavedec(y_50Hz, 'db8', level=maxlev)
#             for n in range(1, len(coffs)):
#                 coffs[n] = pywt.threshold(coffs[n], threshold * max(abs(coffs[n])))
#
#             xiaobo = pywt.waverec(coffs, 'db8')
#             processed_channel_data = xiaobo[1:]
#
#             cyy_2d = processed_channel_data.reshape(-1, 1)
#             data_normalized = scaler.fit_transform(cyy_2d)
#             data_normalized1 = data_normalized.reshape(-1)
#
#             preprocessed_data.append(data_normalized1)
#
#             if new_length is None:
#                 new_length = len(processed_channel_data)
#             else:
#                 new_length = min(new_length, len(processed_channel_data))
#
#         preprocessed_data = [channel[:new_length] for channel in preprocessed_data]
#
#         # 将每个通道的数据合并为一个二维数组
#         preprocessed_data = np.column_stack(preprocessed_data)
#
#         processed_data_all.append(preprocessed_data)
#
#     new_var_name = "Datanew" + str(i) # 构建新变量名字符串
#     processed_data_all = np.array(processed_data_all)
#     globals()[new_var_name] = processed_data_all
#
#     scio.savemat('D:/sEMG_0909/data/precesseddata/S' + str(i) + '_data.mat',
#                  {'data': processed_data_all, 'label': label_lin})
#
#






















