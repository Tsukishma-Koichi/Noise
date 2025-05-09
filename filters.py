import numpy as np
import librosa
from scipy.signal import iirnotch, lfilter, lfilter_zi, spectrogram
from sklearn.preprocessing import minmax_scale

# ----------------------
# 新增滤波模块
# ----------------------
class AudioFilter:
    @staticmethod
    def _find_main_frequency(y, sr):
        """辅助函数：定位主噪声频率"""
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        spectral_avg = np.mean(S, axis=1)
        main_freq = freqs[np.argmax(spectral_avg)]
        return main_freq

    @staticmethod
    def iir_notch_filter(y, sr, Q=30):
        """IIR陷波滤波器（用于稳态噪声）"""
        main_freq = AudioFilter._find_main_frequency(y, sr)
        w0 = main_freq / (sr / 2)
        b, a = iirnotch(w0, Q)
        zi = lfilter_zi(b, a)
        y_filtered, _ = lfilter(b, a, y, zi=zi * y[0])
        return y_filtered

    @staticmethod
    def lms_adaptive_filter(y, sr, filter_order=128, mu=0.1):
        """LMS自适应滤波器（用于非稳态噪声）"""
        # 生成参考噪声（基于频谱分析）
        n = len(y)
        t = np.arange(n) / sr

        # 生成带限噪声作为参考
        _, _, Sxx = spectrogram(y, sr)
        f_peaks = np.mean(Sxx, axis=1)
        peak_freqs = np.argsort(f_peaks)[-2:] * (sr / 2) / (len(f_peaks) - 1)
        noise = np.sum([np.sin(2 * np.pi * f * t) for f in peak_freqs], axis=0)
        noise = minmax_scale(noise, feature_range=(-0.1, 0.1))

        # LMS滤波
        w = np.zeros(filter_order)
        y_filtered = np.zeros_like(y)

        for i in range(filter_order, len(y)):
            x = noise[i - filter_order:i]
            y_filtered[i] = np.dot(w, x)
            e = y[i] - y_filtered[i]
            w += mu * e * x

        return y_filtered