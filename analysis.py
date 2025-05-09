import librosa
import numpy as np
from scipy.signal import find_peaks


# ----------------------
# 音频分析模块
# ----------------------
def analyze_audio(filename):
    """执行音频特征分析（已修复类型转换问题）"""
    try:
        y, sr = librosa.load(filename, sr=None, mono=True)
    except Exception as e:
        raise RuntimeError(f"音频加载失败: {str(e)}")

    # STFT参数
    n_fft = 2048
    hop_length = 512

    # 计算频谱
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 特征提取（带数值稳定性处理）
    energy_changes = []
    for i in range(magnitude.shape[1]):
        frame = magnitude[:, i]
        if i > 0:
            prev_energy = np.sum(magnitude[:, i - 1] ** 2) + 1e-12
            curr_energy = np.sum(frame ** 2)
            db_change = 20 * np.log10(curr_energy / prev_energy)
            energy_changes.append(float(db_change))
        else:
            energy_changes.append(0.0)

    # 计算频谱特征
    spectral_features = _calculate_spectral_features(np.mean(magnitude, axis=1), freqs)

    # 特征工程（显式类型转换）
    features = {
        "bandwidth_3db": float(spectral_features["bandwidth_3db"]),
        "peak_energy_ratio": float(spectral_features["peak_energy_ratio"]),
        "max_energy_change": float(np.max(np.abs(energy_changes))),
        "energy_change_std": float(np.std(energy_changes)),
        "spectral_flatness": float(np.mean(librosa.feature.spectral_flatness(S=magnitude)))
    }

    # 数值范围验证
    _validate_features(features)
    return features


def _calculate_spectral_features(spectral_avg, freqs):
    """计算频谱特征（带边界保护）"""
    # 找主峰（处理无峰情况）
    peaks, _ = find_peaks(spectral_avg, height=0.5 * np.max(spectral_avg))
    main_peak_idx = peaks[0] if len(peaks) > 0 else np.argmax(spectral_avg)

    # 计算3dB带宽（防止越界）
    peak_value = spectral_avg[main_peak_idx]
    threshold = peak_value * 0.707  # -3dB
    lower = upper = main_peak_idx

    while lower > 0 and spectral_avg[lower] >= threshold:
        lower -= 1
    lower = max(lower, 0)

    while upper < len(spectral_avg) - 1 and spectral_avg[upper] >= threshold:
        upper += 1
    upper = min(upper, len(spectral_avg) - 1)

    # 计算主峰能量占比（数值安全处理）
    total_energy = np.sum(spectral_avg ** 2) + 1e-12
    peak_energy = np.sum(spectral_avg[lower:upper + 1] ** 2)

    return {
        "bandwidth_3db": float(freqs[upper] - freqs[lower]),
        "peak_energy_ratio": float(peak_energy / total_energy),
        "lower": int(lower),
        "upper": int(upper)
    }


def _validate_features(features):
    """特征值验证（防御性编程）"""
    if not 0 <= features["peak_energy_ratio"] <= 1:
        raise ValueError(f"无效能量占比值: {features['peak_energy_ratio']}")
    if features["bandwidth_3db"] < 0:
        raise ValueError(f"负带宽值: {features['bandwidth_3db']}")