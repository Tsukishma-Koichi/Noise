# visualization.py
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


class AudioVisualizer:
    @staticmethod
    def plot_spectrogram(signal, sr, filename):
        """
        生成无标题的语谱图
        参数：
            signal (np.array): 音频信号数组
            sr (int): 采样率
            filename (str): 输出文件基础路径（无需扩展名）
        """
        # 参数校验
        if not isinstance(signal, np.ndarray):
            raise ValueError("信号必须为numpy数组")
        if sr <= 0:
            raise ValueError("采样率必须为正整数")

        plt.figure(figsize=(12, 4))

        # 计算STFT
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(signal)),
            ref=np.max
        )

        # 绘制语谱图
        librosa.display.specshow(
            D,
            sr=sr,
            x_axis='time',
            y_axis='log',
            cmap='viridis'
        )

        # 隐藏标题（关键修正点）
        plt.gca().set_title('')

        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

        # 保存文件（自动添加后缀）
        plt.savefig(f"{filename}_spectrogram.png",
                    dpi=150,
                    bbox_inches='tight',
                    pad_inches=0.2)
        plt.close()

    @staticmethod
    def plot_wave_comparison(original, processed, sr, filename):
        """
        生成无标题的波形对比图
        参数：
            original (np.array): 原始信号
            processed (np.array): 处理后的信号
            sr (int): 采样率
            filename (str): 输出文件基础路径
        """
        # 参数校验
        if len(original) != len(processed):
            raise ValueError("信号长度不一致")

        plt.figure(figsize=(12, 6))
        t = np.arange(len(original)) / sr

        # 绘制双波形
        plt.plot(t, original,
                 alpha=0.6,
                 linewidth=0.8,
                 color='#1f77b4',
                 label='原始信号')

        plt.plot(t, processed,
                 alpha=0.8,
                 linewidth=1.2,
                 color='#ff7f0e',
                 label='降噪信号')

        # 隐藏标题（关键修正点）
        plt.gca().set_title('')

        plt.legend(loc='upper right', fontsize=8)
        plt.grid(True, alpha=0.2)
        plt.xlabel('时间 (秒)', fontsize=9)
        plt.ylabel('归一化振幅', fontsize=9)
        plt.tight_layout()

        plt.savefig(f"{filename}_wave_compare.png",
                    dpi=150,
                    bbox_inches='tight')
        plt.close()