from IPython.display import display
from utils import load_data, db6_wavelet_denoise,load_heartsound_features
import IPython.display as ipd   # 用于播放音频
import librosa.display  # 用于绘制音频波形图和频谱图
import librosa  # 音频处理库
import matplotlib.pyplot as plt
import numpy as np

# 加载标签为高质量和低质量的原始心音数据
dataset_arr = load_data()
high_quality_dataset = dataset_arr[dataset_arr[:, 1] == 'high-quality']
low_quality_dataset = dataset_arr[dataset_arr[:, 1] == 'low-quality']

# 加载提取好的特征，包括完整、高质量和低质量部分
hs_df, high_quality_df, low_quality_df = load_heartsound_features()

# 从数据集中随机选择样本
def _get_random_samples(n_samples, seed):
    if(n_samples > 10 or seed > 500):
        raise ValueError("Max samples is 10 and is even, seed max is 500")

    np.random.seed(seed)
    a = np.random.randint(0, 500)

    high_quality_samples = [
        (high_quality_dataset[a+i][0], f"High-quality Sample {i}") for i in range(n_samples)]
    low_quality_samples = [
        (low_quality_dataset[a+i][0], f"Low-quality Sample {i}") for i in range(n_samples)]
    sample_sounds = high_quality_samples + low_quality_samples
    return sample_sounds

# 播放随机选择的样本，并显示其波形图和频谱图
def view_random_samples(n_samples=2, seed=0):
    sample_sounds = _get_random_samples(n_samples, seed)
    for sound in sample_sounds:
        sample_sound, title = sound
        _, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
        librosa.display.waveshow(sample_sound, sr=2000, ax=axs[0])
        sound_stft = librosa.stft(sample_sound)
        sound_spectrogram = librosa.amplitude_to_db(abs(sound_stft))
        librosa.display.specshow(
            sound_spectrogram, sr=2000, x_axis='time', y_axis='log', ax=axs[1])
        axs[0].set_title(title)

        plt.show()
        display(ipd.Audio(sample_sound, rate=2000))

# 显示一个低质量心音样本及其小波去噪后的效果,绘制波形图和频谱图进行对比
def view_wavlet_denoising():
    low_quality_sound = _get_random_samples(1, 0)[1][0]
    sample_sounds = [low_quality_sound, db6_wavelet_denoise(low_quality_sound)]
    labels = ["Low-quality", "Low-quality Denoised"]
    for i, sound in enumerate(sample_sounds):
        sample_sound, title = sound, labels[i]
        _, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 3))
        librosa.display.waveshow(sample_sound[:2000], sr=2000, ax=axs[0])
        X = librosa.stft(sample_sound)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(
            Xdb, sr=2000, x_axis='time', y_axis='log', ax=axs[1])
        axs[0].set_title(title)
        plt.show()