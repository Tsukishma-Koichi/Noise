import os
from datetime import datetime
import time
import json
import traceback
import re
import numpy as np
import sounddevice as sd
import soundfile as sf
import keyboard
from scipy.signal import iirnotch, lfilter, firwin
from scipy.signal.windows import hann
from sklearn.preprocessing import minmax_scale
from scipy.signal import lfilter_zi
from scipy.signal import filtfilt
from sklearn.cluster import KMeans
from scipy.signal import spectrogram
from scipy.signal import find_peaks
from dotenv import load_dotenv
import librosa
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统中文支持
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载环境变量
load_dotenv()

class AudioVisualizer:
    @staticmethod
    def plot_spectrogram(signal, sr, filename, title):
        """修复中文标题显示"""
        plt.figure(figsize=(10, 6))
        # ...原有处理逻辑...
        plt.title(title, fontproperties='SimHei')  # 指定中文字体
        plt.savefig(f"{filename}_spectrogram.png", dpi=150, bbox_inches='tight')
        plt.close()


# ----------------------
# 新增录音模块
# ----------------------
def record_audio():
    """实时音频采集（采样率16000Hz，单声道）"""
    fs = 16000  # 与后续分析匹配的采样率
    channels = 1
    filename = f"recording_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"

    print("\n麦克风准备中...（请允许浏览器权限提示）")
    try:
        # 初始化录音缓冲区
        recorded_data = []

        # 定义录音回调函数
        def audio_callback(indata, frames, time, status):
            recorded_data.append(indata.copy())

        # 开始录音
        with sd.InputStream(samplerate=fs, channels=channels, callback=audio_callback):
            print("录音开始，按下键盘P键结束采集...")
            while True:
                if keyboard.is_pressed('p'):  # 检测P键按下
                    print("检测到停止指令，正在保存文件...")
                    break
                time.sleep(0.1)

        # 合并音频数据
        audio_array = np.concatenate(recorded_data, axis=0)

        # 保存为WAV文件
        sf.write(filename, audio_array, fs)
        print(f"音频已保存至：{os.path.abspath(filename)}")
        return filename

    except Exception as e:
        raise RuntimeError(f"录音失败: {str(e)}")

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


# ----------------------
# 智能分类模块
# ----------------------
class NoiseClassifier:
    def __init__(self, model="deepseek-chat"):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.model = model
        self.local_rules = [
            (self._is_steady,
             {"classification": "steady", "confidence": 0.90, "reason": "窄带且能量稳定"}),
            (self._is_non_steady,
             {"classification": "non-steady", "confidence": 0.85, "reason": "检测到时变特征"})
        ]
        self.params = {
            "bandwidth_threshold": 50,
            "energy_std_threshold": 3,
            "spike_threshold": 15
        }

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type((Exception)))
    def classify(self, features):
        """分类入口（带请求预验证）"""
        try:
            self._prevalidate_request(features)
            return self._api_classify(features)
        except Exception as e:
            self._log_error(e)
            print(f"API分类失败，启用本地规则: {str(e)}")
            return self._local_classify(features)

    def _prevalidate_request(self, features):
        """请求预处理（类型检查）"""
        try:
            json.dumps(features)  # 测试序列化
        except TypeError as e:
            type_info = {k: str(type(v)) for k, v in features.items()}
            raise RuntimeError(f"特征值类型错误: {type_info}") from e

    def _api_classify(self, features):
        """API分类（增强提示工程）"""
        prompt = f"""
作为声学专家，请根据以下特征进行噪声分类（需严格遵循标准）：
{json.dumps(features, indent=2, ensure_ascii=False)}

分类标准（必须逐条检查）：
1. 稳态噪声需同时满足：
   - 主峰3dB带宽 <{self.params['bandwidth_threshold']}Hz
   - 主峰能量占比 >70%
   - 能量波动标准差 <{self.params['energy_std_threshold']}dB

2. 非稳态噪声需满足任一：
   - 存在超过{self.params['spike_threshold']}dB的瞬时能量突增
   - 主峰带宽 ≥{self.params['bandwidth_threshold'] * 2}Hz

请返回严格符合以下格式的JSON：
{{
    "classification": "steady/non-steady",
    "confidence": 置信度(0-1),
    "reason": "技术说明（需引用具体参数）"
}}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个严谨的噪声分类引擎，必须严格遵循技术标准"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )

        return self._process_response(response, features)

    def _process_response(self, response, features):
        """响应处理（正则表达式增强）"""
        content = response.choices[0].message.content

        # 使用正则表达式提取JSON内容
        if match := re.search(r'```(?:json)?\n?(.*?)```', content, re.DOTALL):
            content = match.group(1).strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"无效的JSON响应: {content}") from e

        return self._validate_result(result, features)

    def _validate_result(self, result, features):
        """结果验证（增强逻辑检查）"""
        valid_classes = ["steady", "non-steady"]
        if result["classification"] not in valid_classes:
            raise ValueError(f"无效分类结果: {result['classification']}")

        # 稳态判定逻辑验证
        if result["classification"] == "steady":
            failed_conditions = []
            if features["bandwidth_3db"] >= self.params['bandwidth_threshold']:
                failed_conditions.append(f"带宽{features['bandwidth_3db']}≥阈值{self.params['bandwidth_threshold']}")
            if features["peak_energy_ratio"] < 0.7:
                failed_conditions.append(f"能量占比{features['peak_energy_ratio'] * 100:.1f}%<70%")
            if features["energy_change_std"] >= self.params['energy_std_threshold']:
                failed_conditions.append(
                    f"波动标准差{features['energy_change_std']:.1f}≥阈值{self.params['energy_std_threshold']}")

            if failed_conditions:
                raise ValueError(f"稳态判定矛盾: {', '.join(failed_conditions)}")

        return result

    def _is_steady(self, features):
        """稳态本地规则"""
        return (features["bandwidth_3db"] < self.params['bandwidth_threshold'] and
                features["peak_energy_ratio"] > 0.7 and
                features["energy_change_std"] < self.params['energy_std_threshold'])

    def _is_non_steady(self, features):
        """非稳态本地规则"""
        return (features["max_energy_change"] > self.params['spike_threshold'] or
                features["bandwidth_3db"] >= self.params['bandwidth_threshold'] * 2)

    def _local_classify(self, features):
        """本地分类（带阈值检查）"""
        for condition, result in self.local_rules:
            if condition(features):
                return result
        return {"classification": "unknown", "confidence": 0.0, "reason": "未匹配任何规则"}

    def _log_error(self, error):
        """增强错误日志"""
        log_entry = f"""
[{datetime.now().isoformat()}] 错误详情
类型: {type(error).__name__}
信息: {str(error)}
追踪:
{''.join(traceback.format_exception(type(error), error, error.__traceback__))}
        """
        with open("classification_errors.log", "a", encoding="utf-8") as f:
            f.write(log_entry)


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


# ----------------------
# 修改后的处理流程
# ----------------------
def process_audio(filename, classification):
    """增强处理流程（包含可视化）"""
    try:
        # 获取干净文件名
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_dir = "processed"
        os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
        # 加载原始音频
        y, sr = librosa.load(filename, sr=None, mono=True)
        base_name = os.path.splitext(filename)[0]

        # 显示原始语谱图
        AudioVisualizer.plot_spectrogram(y, sr, base_name, "原始语谱图")

        # 执行降噪
        if classification == "steady":
            print("应用IIR陷波滤波器...")
            processed = AudioFilter.iir_notch_filter(y, sr)
        else:
            print("应用LMS自适应滤波器...")
            processed = AudioFilter.lms_adaptive_filter(y, sr)

        # 限制幅值范围
        processed = np.clip(processed, -1.0, 1.0)

        # 保存音频
        output_path = f"processed_{os.path.basename(filename)}"
        sf.write(output_path, processed, sr)
        print(f"降噪音频已保存至：{os.path.abspath(output_path)}")

        # 生成可视化结果到processed目录
        print("生成分析图表...")
        plot_base_path = os.path.join(output_dir, base_name)
        AudioVisualizer.plot_wave_comparison(y, processed, sr, plot_base_path)
        AudioVisualizer.plot_spectrogram(processed, sr, plot_base_path, "降噪后语谱图")
        AudioVisualizer.plot_snr_curve(y, processed, sr, plot_base_path)

    except Exception as e:
        raise RuntimeError(f"音频处理失败: {str(e)}")


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

# ----------------------
# 增强版主程序
# ----------------------
# ...（保持之前的所有import和类定义不变）...

if __name__ == "__main__":
    print("""\n
    =============================
      智能降噪系统 V3.1
    =============================
    """)

    try:
        # 模式选择
        while True:
            try:
                choice = int(input("请选择输入方式：\n1. 本地文件分析\n2. 实时录音分析\n请输入选项 (1/2): "))
                if choice in (1, 2):
                    break
                print("请输入有效的选项数字")
            except ValueError:
                print("输入错误，请重新输入")

        # 获取文件路径（修复path变量定义问题）
        if choice == 1:
            while True:
                file_path = input("请输入音频文件路径：").strip()  # 变量名改为file_path
                if os.path.exists(file_path):
                    break
                print(f"文件未找到: {file_path}")
        else:
            file_path = record_audio()

        # 执行分析
        features = analyze_audio(file_path)
        print("\n=== 特征分析结果 ===")
        for k, v in features.items():
            print(f"{k:20}: {v:.2f}")

        # 分类处理
        classifier = NoiseClassifier()
        result = classifier.classify(features)

        # 打印结果
        print("\n=== 分类报告 ===")
        print(f"噪声类型: {result['classification'].upper()}")
        print(f"置信度: {result['confidence'] * 100:.1f}%")
        print(f"技术分析: {result['reason']}")

        # 新增音频处理流程（使用正确的变量名）
        process_audio(file_path, result['classification'])  # 改为file_path变量

    except Exception as e:
        print(f"\n严重错误: {str(e)}")
        traceback.print_exc()