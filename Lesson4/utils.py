import numba
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pywt
import math

# 打印当前工作目录
print("当前工作目录：", os.getcwd())
# 设置项目根目录(修改为当前工作路径)
BASE_DIR = os.chdir(r'D:\Code_Files\实验4\code')#os.chdir用于切换到指定目录

# 读取一条音频记录并将其数据和标签添加到sounds列表中
@numba.jit(forceobj=True)
def add_sound(curr_row, sounds):
    _, path, label = tuple(curr_row)
    _, sound = wavfile.read(path)
    sound = sound.astype(np.float32)
    sounds.append([sound, label])
    return curr_row

# 加载数据集中所有音频文件，生成包含音频数组和标签的numpy数组
def _load_dataset_array(dataset_df):
    sounds = []
    dataset_df.apply(lambda row: add_sound(row, sounds), axis=1, raw=True)
    return np.array(sounds, dtype=object)

# 绘制音频波形图
def plot_hs(path, x=[], dur=1.5, verts=[], sr=4000):
    y, sr = librosa.load(path, sr=sr, duration=dur) if len(x) == 0 else (x, sr)
    _, ax = plt.subplots(nrows=1, figsize=(20, 4))
    for line in verts:
        ax.axvline(x=line, color="red", ls='--')
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title=f'{path.split("/")[-1]}')
    plt.show()
    return (y, sr)

# 信号归一化，默认方法为min-max
def normalize(signal, mode="min-max"):
    base = min(signal) if mode == "min-max" else 0
    range_val = max(signal) - base
    normalized = [(x - base) / range_val for x in signal]
    return np.array(normalized)

# 读取dataset文件夹中的RECORDS-high-quality或RECORDS-low-quality文件，返回包含文件名、路径、标签的列表
def load_dataset(label):
    record_file = f'dataset{os.sep}RECORDS-{label}'
    with open(record_file, 'r') as f:
        lines = f.readlines()
    records = []
    for line in lines:
        name = line.strip().replace('.wav', '')
        wav_path = f'dataset{os.sep}{name}.wav'
        records.append([name, wav_path, label])
    return records


# 构建整个数据集dataframe,包含所有高质量与低质量的心音文件记录
def build_dataframe():
    records = []
    records.extend(load_dataset('high-quality'))
    records.extend(load_dataset('low-quality'))
    return pd.DataFrame(records, columns=["name", "path", "label"])

# 加载完整的心音数据集及其标签,
def load_data():
    dataframe = build_dataframe()
    sounds = _load_dataset_array(dataframe)
    return sounds

# 小波降噪处理（使用d6小波）
def db6_wavelet_denoise(x):
    a5, d5, d4, d3, d2, d1 = pywt.wavedec(x, 'db6', level=5)
    reconstructed = pywt.waverec([a5, d5, np.zeros_like(d4), d3, d2, np.zeros_like(d1)], 'db6')
    return reconstructed

# 特征分布直方图绘图
def features_histo(dfs, labels=None):
    features_names = dfs[0].columns
    fig, axs = plt.subplots(nrows=15, ncols=3, figsize=(20, 40), gridspec_kw={'hspace': 1.2})
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, feature in enumerate(features_names):
        ax = axs[math.floor(i / 3)][i % 3]
        for j, df in enumerate(dfs):
            ax.hist(df[feature], bins=30, alpha=0.6, label=labels[j] if labels else None,
                    color=colors[j % len(colors)], edgecolor='black')
        ax.set_title(feature)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    if labels:
        axs[0][0].legend()
    plt.tight_layout()
    plt.show()

# 读取保存特征标签的CSV文件，并进行按标签划分
def load_heartsound_features():
    hs_df = pd.read_csv("dataset.csv", encoding='utf-8')
    hs_df = hs_df.sort_values(by='Label', ascending=False)

    high_quality_df = hs_df[hs_df["Label"] == "high-quality"]
    low_quality_df = hs_df[hs_df["Label"] == "low-quality"]
    return hs_df, high_quality_df, low_quality_df