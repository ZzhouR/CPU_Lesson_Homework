import numpy as np
import scipy.stats
import scipy.signal
import math
import librosa
import pandas as pd
import numpy as np
import utils
import segmentation
from tqdm import tqdm
import antropy as ant
from scipy.signal import decimate

# 获取信号最大振幅
def get_max_amplitude(signal):
    return max(list(signal))

# 获取主导频率
def get_dominant_frequency(signal,sampling_rate =4000):
    fourier = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(np.array(signal).size, d=1.0/sampling_rate) 
    positive_frequencies = frequencies[np.where(frequencies >= 0)] 
    magnitudes = abs(fourier[np.where(frequencies >= 0)])
    peak_frequency = np.argmax(magnitudes)
    return positive_frequencies[peak_frequency]

# 计算熵值
def get_entropy(signal):
    signal = np.array(signal)
    if not np.isfinite(signal).all():
        return 0.0  
    number_of_bins = max(10, math.ceil(math.sqrt(len(signal))))
    counts, _ = np.histogram(signal, bins=number_of_bins)
    counts = counts + 1e-12  
    return scipy.stats.entropy(counts)

# 提取统计特征：均值、中位数、标准差、峰度、偏度、四分位间距、百分位数
def stat_features(array):
    mean= np.mean(array)
    median = np.median(array)
    std= np.std(array)
    kurtosis = scipy.stats.kurtosis(array, axis=0, bias=True)
    skewness = scipy.stats.skew(array, axis=0, bias=True)
    iqr = scipy.stats.iqr(array)
    first_percentile = np.percentile(array,25)
    second_percentile = np.percentile(array,50)
    third_percentile = np.percentile(array,75)
    return mean,median,std,kurtosis,skewness,iqr,first_percentile,second_percentile,third_percentile

# 提取13维MFCC特征
def mfcc(array,sampling_rate=4000):
    mfccs = librosa.feature.mfcc(y=array, sr=sampling_rate,n_mfcc=13)
    mfccs = np.mean(mfccs,axis=1)
    return list(mfccs.flatten())

# 零交叉率：信号过零点的频率
def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)

# 均方根值
def rms(signal):
    return np.sqrt(np.mean(signal**2))

# 波形因子 = 均方根 / 平均绝对值，衡量波形形状
def waveform_factor(signal):
    return safe_divide(rms(signal), np.mean(np.abs(signal)))

# 脉冲因子 = 最大绝对值 / 平均绝对值，反映脉冲峰值特性
def pulse_factor(signal):
    return safe_divide(np.max(np.abs(signal)), np.mean(np.abs(signal)))

# 裕度因子 = 最大绝对值 / 均方根，衡量信号峰值尖锐度
def margin_factor(signal):
    return safe_divide(np.max(np.abs(signal)), rms(signal))

# 频谱质心
def spectral_centroid(signal, sampling_rate=4000):
    return librosa.feature.spectral_centroid(y=signal, sr=sampling_rate).mean()

# 频谱带宽：频谱分布宽度
def spectral_bandwidth(signal, sampling_rate=4000):
    return librosa.feature.spectral_bandwidth(y=signal, sr=sampling_rate).mean()

# 谱平坦度：衡量频谱是否平滑
def spectral_flatness(signal):
    return librosa.feature.spectral_flatness(y=signal).mean()

# 频谱滚降点，表示频谱中某百分比能量的位置
def spectral_rolloff(signal, sampling_rate=4000, roll_percent=0.85):
    return librosa.feature.spectral_rolloff(y=signal, sr=sampling_rate, roll_percent=roll_percent).mean()

# 频谱峰度：高阶统计量，衡量尖锐性
def spectral_kurtosis(signal, sampling_rate=4000):
    S = np.abs(librosa.stft(signal))
    sum_S = np.sum(S, axis=0, keepdims=True)
    sum_S[sum_S == 0] = 1e-12  # 避免除以0
    S = S / sum_S
    mean_spec = np.sum(S * np.arange(S.shape[0])[:, None], axis=0)
    variance = np.sum(S * (np.arange(S.shape[0])[:, None] - mean_spec)**2, axis=0)
    kurtosis = np.sum(S * (np.arange(S.shape[0])[:, None] - mean_spec)**4, axis=0) / (variance**2 + 1e-12)
    return np.mean(kurtosis)

# 样本熵：衡量信号复杂度
def sample_entropy(segment, sr):
    try:
        # 降采样
        if sr > 500:
            segment = decimate(segment, sr // 500)
        if len(segment) > 2000:
            mid = len(segment) // 2
            segment = segment[mid - 1000: mid + 1000]
        sampen = ant.sample_entropy(segment)
    except Exception:
        sampen = np.nan
    return sampen

# 近似熵
def approximate_entropy(segment, sr):
    try:
        if sr > 500:
            segment = decimate(segment, sr // 500)
        if len(segment) > 2000:
            mid = len(segment) // 2
            segment = segment[mid - 1000: mid + 1000]
        apen = ant.app_entropy(segment)
    except Exception:
        apen = np.nan
    return apen

# 避免除以0的安全除法
def safe_divide(a, b, eps=1e-12):
    b = np.where(np.abs(b) < eps, eps, b)
    return a / b

# 计算 Hjorth 参数：活动性(activity)，移动性(mobility)，复杂性(complexity)
def hjorth_parameters(signal):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    var_zero = np.var(signal)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    activity = var_zero
    mobility = np.sqrt(safe_divide(var_d1, var_zero))
    complexity = np.sqrt(safe_divide(var_d2, var_d1)) / (mobility + 1e-12)

    return activity, mobility, complexity

def envelope_features(signal, sampling_rate=4000):
    # 利用希尔伯特变换计算包络
    analytic_signal = scipy.signal.hilbert(signal)
    envelope = np.abs(analytic_signal)
    mean_env = np.mean(envelope)    # 包络均值
    std_env = np.std(envelope)    # 包络标准值
    max_env = np.max(envelope)    # 包络最大值
    f, Pxx = scipy.signal.welch(envelope, fs=sampling_rate)
    total_power = np.sum(Pxx)
    low_freq_power = np.sum(Pxx[f <= 50]) 
    energy_ratio = low_freq_power / (total_power + 1e-12)   # 包络低频能量比
    return mean_env, std_env, max_env, energy_ratio

# 总特征提取函数
def extract_segment_features(segment, sampling_rate=4000):
    segment = np.array(segment)
    max_val = np.max(np.abs(segment))
    if max_val > 0:
        scale_factor = 50000.0 / max_val
        segment = segment * scale_factor

    features = []

    # ————时域特征————
    features.append(get_max_amplitude(segment))
    mean, median, std, kurtosis, skewness, iqr, p25, p50, p75 = stat_features(segment)
    features.extend([mean, median, std, kurtosis, skewness, iqr, p25, p50, p75])
    features.append(zero_crossing_rate(segment))
    features.append(rms(segment))
    features.append(waveform_factor(segment))
    features.append(pulse_factor(segment))
    features.append(margin_factor(segment))

    # ————频域特征————
    features.append(get_dominant_frequency(segment, sampling_rate))
    features.append(spectral_centroid(segment, sampling_rate))
    features.append(spectral_bandwidth(segment, sampling_rate))
    features.append(spectral_flatness(segment))
    features.append(spectral_rolloff(segment, sampling_rate))
    features.append(spectral_kurtosis(segment, sampling_rate))

    # ————非线性特征————
    features.append(get_entropy(segment))
    
    # 这里调用你的 sample_entropy 和 approximate_entropy 函数，传入 segment 和 sampling_rate
    features.append(sample_entropy(segment, sampling_rate))
    features.append(approximate_entropy(segment, sampling_rate))
    
    activity, mobility, complexity = hjorth_parameters(segment)
    features.extend([activity, mobility, complexity])

    # ————包络特征————
    mean_env, std_env, max_env, energy_ratio = envelope_features(segment, sampling_rate)
    features.extend([mean_env, std_env, max_env, energy_ratio])

    # ————MFCC特征————
    features.extend(mfcc(segment, sampling_rate))

    return features

# 根据所有 segments 提取特征，并构建特征矩阵列表
def build_features_df(segments, sampling_rate=4000):
    data, labels = separate_labels(segments)
    dataframe = []
    for index, element in enumerate(tqdm(data, desc="Extracting features")):
        features = extract_segment_features(element, sampling_rate)
        features.append(labels[index])
        dataframe.append(features)
    return dataframe

# 将segments划分为音频信号数据和对应标签两个列表
def separate_labels(segments):
    data = []
    labels = []
    for row in segments:
        data.append(row[0])
        labels.append(row[1])
    return data, labels

# 从原始数据构建完整的特征dataframe并保存为CSV文件
def construct_dataframe(dataset_name, sampling_rate=4000):
    records = utils.load_data()
    segments = segmentation.build_segements(records, sr=sampling_rate)

    features_matrix = build_features_df(segments, sampling_rate)

    columns = [
        # 时域特征
        "Max_Amplitude", "Mean", "Median", "STD", "Kurtosis", "Skewness", "IQR",
        "First_Percentile", "Second_Percentile", "Third_Percentile",
        "Zero_Crossing_Rate", "RMS", "Waveform_Factor", "Pulse_Factor", "Margin_Factor",

        # 频域特征
        "Dominant_Freq", "Spectral_Centroid", "Spectral_Bandwidth",
        "Spectral_Flatness", "Spectral_Rolloff", "Spectral_Kurtosis",

        # 非线性特征
        "Entropy", "Sample_Entropy", "Approximate_Entropy",
        "Hjorth_Activity", "Hjorth_Mobility", "Hjorth_Complexity",

        # 包络特征
        "Envelope_Mean", "Envelope_STD", "Envelope_Max", "Envelope_Energy_Ratio",

        # MFCC特征13维
        "MFCC1", "MFCC2", "MFCC3", "MFCC4", "MFCC5", "MFCC6", "MFCC7",
        "MFCC8", "MFCC9", "MFCC10", "MFCC11", "MFCC12", "MFCC13",

        "Label"
    ]

    dataframe = pd.DataFrame(features_matrix, columns=columns)
    # 过滤掉最大振幅为0的异常数据
    dataframe = dataframe[dataframe["Max_Amplitude"] != 0]
    dataframe.to_csv(f"{dataset_name}.csv", index=False)
    return dataframe

if __name__ == "__main__":
    construct_dataframe("dataset")