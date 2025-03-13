import numpy as np
import pandas as pd
import datetime
import math
import os
import glob
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as scio

# 信号处理相关库
from scipy import signal
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.signal import filtfilt, iirnotch, freqz, butter, welch, sosfilt
import pywt

# 数据预处理
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

# 机器学习相关库
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier



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

def calculate_iEMG(data):
    """
    计算给定数据的iEMG（积分肌电图）
    
    参数：
    data: 包含数据的列表或数组
    
    返回值：
    iEMG: 计算得到的积分肌电图值
    """
    # 确保输入为NumPy数组
    data = np.asarray(data)
    
    # 计算iEMG
    iEMG = np.sum(np.abs(data))
    
    return iEMG

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
def integratedEMG(X):
    iEMG=X.flatten()
    iEMG = np.abs(iEMG)
    iEMG = np.sum(iEMG)
    return iEMG

def mean_power_frequency(signal, fs):
    """
    计算信号的均方功率频率（Mean Power Frequency, MPF）

    参数:
    - signal: 输入信号（numpy 数组）
    - fs: 采样频率（Hz）

    返回:
    - MPF: 均方功率频率（Hz）
    """
    # 计算功率谱密度
    f, Pxx = welch(signal, fs=fs)
    
    # 计算均方功率频率
    mpf = np.sum(f * Pxx) / np.sum(Pxx)
    
    return mpf


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

scaler = MinMaxScaler(feature_range=(-1, 1))

for i in range(1,4):

    file_name = fr'D:\SEMG\sEMGdata\data\split\S2-{i}_data2.mat'
    mat_data = scio.loadmat(file_name)

    data_data1= mat_data['subset1_data']
    data_data2= mat_data['subset2_data']
    data_data= np.vstack((data_data1,data_data2))
    data_label1= mat_data['subset1_label']
    data_label2= mat_data['subset2_label']
    data_label= np.vstack((data_label1,data_label2))




    processed_data_all = []


    new_length = None
    for j in range(len(data_data)):
        datalin0 = data_data[j][0]

        preprocessed_data = []
        new_length = None

        for channel in range(8):
        
            channel_data = datalin0[:, channel]
            
            # 低通滤波
            sos = signal.butter(30, 499, btype='low', output='sos', fs=1000)
            LF_x1 = sosfilt(sos, channel_data)

            # Notch 滤波
            fs = 1000
            f0 = 50
            w0 = f0 / (fs / 2)
            Q = 30
            b, a = iirnotch(w0, Q)
            y_50Hz = filtfilt(b, a, LF_x1)

            # 数据归一化
            data_normalized = scaler.fit_transform(y_50Hz.reshape(-1, 1)).flatten()
            preprocessed_data.append(data_normalized)

            # 更新新长度
            if new_length is None:
                new_length = len(data_normalized)
            else:
                new_length = min(new_length, len(data_normalized))

        # 确保每个通道的数据长度一致
        preprocessed_data = [channel[:new_length] for channel in preprocessed_data]

        # 合并通道数据为二维数组
        preprocessed_data = np.column_stack(preprocessed_data)
        processed_data_all.append(preprocessed_data)

    # new_var_name = "Datanew" + str(i)  # 构建新变量名字符串
    processed_data_all = np.array(processed_data_all, dtype=object)
    # globals()[new_var_name] = processed_data_all
    AGdatalin = np.zeros((processed_data_all.shape[0], 6 * 8 * 4))

    for num in range(processed_data_all.shape[0]):
        datalin0 = processed_data_all[num]

        L = len(datalin0) // 6  # 窗长的大小
        d = len(datalin0) % 6  # 需要减去的余数
        datalin0 = datalin0[d:, :]
        RMSj = np.zeros((6, 8))
        iEMGj = np.zeros((6, 8))
        # WLj = np.zeros((6, 8))
        # MAVj = np.zeros((6, 8))
        # SSCj = np.zeros((6, 8))
        # zcj = np.zeros((6, 8))
        mfj = np.zeros((6, 8))
        mpfj = np.zeros((6, 8))
        # powj = np.zeros((6, 8))
        # AACj = np.zeros((6, 8))
        # RMSj = np.zeros((6, 8))
        # MAVj = np.zeros((6, 8))
        for m in range(0, 8):
            linP = datalin0[:, m]
            for n in range(0, 6):
                if n != 5:
                    jisuan = linP[0 + L * n: L + L * (n + 1) + 1]
                    rms = calculate_rms(jisuan)
                    iEMG = calculate_iEMG(jisuan)
                    mpf = mean_power_frequency(jisuan, 1000)
                    # wl = calculate_wl(jisuan)
                    # mav = calculate_mav(jisuan)
                    # ssc = calculate_ssc(jisuan)
                    mf = median_frequency(jisuan, 1000)
                    # pow = power_spectral_entropy(jisuan, 1000)
                    # zc = calculate_zc(jisuan)
                    RMSj[n, m] = rms
                    iEMGj[n, m] = iEMG
                    # WLj[n, m] = wl
                    # MAVj[n, m] = mav
                    # SSCj[n, m] = ssc
                    # ZCj[n, m] = zc
                    mpfj[n, m] = mpf
                    mfj[n, m] = mf
                    # powj[n, m] = pow

                    # AAC = AverageAmptitudeChange(jisuan)
                    # AACj[n, m] = AAC
                    # RMS = RootMeanSquare(jisuan)
                    # RMSj[n, m] = RMS
                    # MAV = ModifiedMeanAbsoluteValue(jisuan)
                    # MAVj[n, m] = MAV
                else:
                    jisuan = linP[L * 5 + 1:]
                    rms = calculate_rms(jisuan)
                    iEMG = calculate_iEMG(jisuan)
                    # wl = calculate_wl(jisuan)
                    # mav = calculate_mav(jisuan)
                    # ssc = calculate_ssc(jisuan)
                    mf = median_frequency(jisuan, 1000)
                    mpf = mean_power_frequency(jisuan, 1000)
                    # pow = power_spectral_entropy(jisuan, 1000)
                    # zc = calculate_zc(jisuan)
                    RMSj[n, m] = rms
                    iEMGj[n, m] = iEMG
                    # WLj[n, m] = wl
                    # MAVj[n, m] = mav
                    # SSCj[n, m] = ssc
                    # ZCj[n, m] = zc
                    mfj[n, m] = mf
                    mpfj[n, m] = mpf
                    # powj[n, m] = pow
                    # AAC = AverageAmptitudeChange(jisuan)
                    # AACj[n, m] = AAC
                    # RMS = RootMeanSquare(jisuan)
                    # RMSj[n, m] = RMS
                    # MAV = ModifiedMeanAbsoluteValue(jisuan)
                    # MAVj[n, m] = MAV
        feature1 = RMSj.flatten('F')
        feature2 = iEMGj.flatten('F')
        # feature2 = WLj.flatten('F')
        # feature3 = MAVj.flatten('F')
        # feature4 = SSCj.flatten('F')
        feature3 = mfj.flatten('F')
        feature4 = mpfj.flatten('F')
        # feature5 = powj.flatten('F')
        
        featurezong = np.append(feature1, feature2)
        featurezong = np.append(featurezong, feature3)
        featurezong = np.append(featurezong, feature4)
        # featurezong = np.append(featurezong, feature5)
        # featurezong = np.append(featurezong, feature6)
        AGdatalin[num, :] = featurezong

    scio.savemat(fr'D:\SEMG\data\prodata\ML6\2-{i}d.mat',
                    {'data': AGdatalin, 'label': data_label})
print('done')

