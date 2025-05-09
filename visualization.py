import matplotlib.pyplot as plt
import numpy as np
import librosa

# ----------------------
# 新增可视化模块
# ----------------------
class AudioVisualizer:
    @staticmethod
    def _safe_normalize(signal):
        """安全归一化到[-1,1]范围"""
        return signal / np.max(np.abs(signal) + 1e-12)

    @staticmethod
    def plot_wave_comparison(original, processed, sr, filename):
        """绘制波形对比图"""
        plt.figure(figsize=(12, 6))

        # 统一时间轴
        t = np.arange(len(original)) / sr
        max_time = t[-1]

        # 绘制原始波形（半透明）
        plt.plot(t, AudioVisualizer._safe_normalize(original),
                 alpha=0.6, label='原始信号', color='blue')

        # 绘制降噪波形（部分透明）
        plt.plot(t, AudioVisualizer._safe_normalize(processed),
                 alpha=0.8, label='降噪信号', color='orange')

        plt.xlabel('时间 (秒)')
        plt.ylabel('归一化幅值')
        plt.title('降噪前后波形对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max_time)

        # 保存并关闭
        plt.savefig(f"{filename}_wave_compare.png", dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_spectrogram(signal, sr, filename, title):
        """绘制语谱图"""
        plt.figure(figsize=(10, 6))

        # 计算STFT
        n_fft = 2048
        hop_length = 512
        S = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)),
                                    ref=np.max)

        # 显示语谱图
        librosa.display.specshow(S, sr=sr, hop_length=hop_length,
                                 x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)

        plt.savefig(f"{filename}_spectrogram.png", dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_snr_curve(original, processed, sr, filename):
        """绘制SNR变化曲线"""
        # 计算噪声分量
        noise = original - processed

        # 分帧计算SNR
        frame_length = 0.1  # 100ms
        samples_per_frame = int(frame_length * sr)

        snr_values = []
        time_points = []

        for i in range(0, len(original), samples_per_frame):
            frame_original = original[i:i + samples_per_frame]
            frame_noise = noise[i:i + samples_per_frame]

            if len(frame_original) < 10:  # 跳过过短帧
                continue

            # 计算功率
            signal_power = np.mean(frame_original ** 2)
            noise_power = np.mean(frame_noise ** 2)

            if noise_power < 1e-12:  # 避免除零
                snr = 50  # 设置上限
            else:
                snr = 10 * np.log10(signal_power / noise_power)
                snr = min(max(snr, -10), 50)  # 限制显示范围

            snr_values.append(snr)
            time_points.append(i / sr)

        # 绘制曲线
        plt.figure(figsize=(12, 4))
        plt.plot(time_points, snr_values, label='瞬时信噪比', color='green')
        plt.axhline(np.mean(snr_values), color='red', linestyle='--',
                    label=f'平均SNR: {np.mean(snr_values):.1f} dB')

        plt.xlabel('时间 (秒)')
        plt.ylabel('SNR (dB)')
        plt.title('信噪比变化曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-10, 55)

        plt.savefig(f"{filename}_snr_curve.png", dpi=150, bbox_inches='tight')
        plt.close()