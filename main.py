# main.py
import os
import time
import traceback
import keyboard
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def clear_matplotlib_cache():
    """清除matplotlib缓存"""
    import matplotlib
    cache_dir = matplotlib.get_cachedir()
    for f in os.listdir(cache_dir):
        if f.endswith(".json"):
            os.remove(os.path.join(cache_dir, f))


def main_menu():
    """显示主菜单"""
    print("""
    =============================
      智能降噪系统 V4.3
    =============================
    """)

    while True:
        try:
            choice = int(input("请选择操作模式：\n1. 文件分析模式\n2. 实时降噪模式\n请输入选项 (1/2): "))
            if choice in (1, 2):
                return choice
            print("请输入有效的选项数字")
        except ValueError:
            print("输入错误，请重新输入")


def file_analysis_mode():
    """文件分析模式入口"""
    from analysis import analyze_audio
    from classifier import NoiseClassifier
    from visualization import AudioVisualizer
    from filters import AudioFilter

    while True:
        file_path = input("请输入音频文件路径：").strip()
        if os.path.exists(file_path):
            break
        print(f"文件未找到: {file_path}")

    try:
        # 特征分析
        features = analyze_audio(file_path)
        print("\n=== 特征分析结果 ===")
        for k, v in features.items():
            print(f"{k:20}: {v:.2f}")

        # 噪声分类
        classifier = NoiseClassifier()
        result = classifier.classify(features)

        # 输出报告
        print("\n=== 分类报告 ===")
        print(f"噪声类型: {result['classification'].upper()}")
        print(f"置信度: {result['confidence'] * 100:.1f}%")
        print(f"技术分析: {result['reason']}")

        # 处理音频
        start_time = time.time()
        process_audio(file_path, result['classification'])
        print(f"处理完成，耗时 {time.time() - start_time:.2f} 秒")

    except Exception as e:
        print(f"处理失败: {str(e)}")
        traceback.print_exc()


def process_audio(filename, classification):
    """增强的音频处理流程"""
    import librosa
    import soundfile as sf
    from visualization import AudioVisualizer
    from filters import AudioFilter

    output_dir = "processed"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 加载音频
        y, sr = librosa.load(filename, sr=None, mono=True)
        base_name = os.path.splitext(os.path.basename(filename))[0]

        # 原始语谱图
        AudioVisualizer.plot_spectrogram(y, sr, os.path.join(output_dir, base_name), "原始语谱图")

        # 执行降噪
        if classification == "steady":
            print("应用IIR陷波滤波器...")
            processed = AudioFilter.iir_notch_filter(y, sr)
        else:
            print("应用LMS自适应滤波器...")
            processed = AudioFilter.lms_adaptive_filter(y, sr)

        # 保存处理结果
        output_path = os.path.join(output_dir, f"processed_{os.path.basename(filename)}")
        sf.write(output_path, processed, sr)
        print(f"降噪音频已保存至：{os.path.abspath(output_path)}")

        # 可视化结果
        plot_base = os.path.join(output_dir, base_name)
        AudioVisualizer.plot_wave_comparison(y, processed, sr, plot_base)
        AudioVisualizer.plot_spectrogram(processed, sr, plot_base, "降噪后语谱图")
        AudioVisualizer.plot_snr_curve(y, processed, sr, plot_base)

    except Exception as e:
        raise RuntimeError(f"音频处理失败: {str(e)}")


def realtime_mode():
    """实时模式入口"""
    import matplotlib
    matplotlib.use('TkAgg')
    from realtime_processor import RealTimeProcessor

    processor = RealTimeProcessor(
        fs=16000,
        block_size=1024,
        update_interval=0.5
    )

    try:
        print("系统初始化完成，正在启动实时降噪...")
        processor.start()  # 此调用将阻塞直到窗口关闭
    except Exception as e:
        print(f"实时处理异常: {str(e)}")
    finally:
        processor.stop()


if __name__ == "__main__":
    clear_matplotlib_cache()  # 清除字体缓存

    try:
        choice = main_menu()
        if choice == 1:
            file_analysis_mode()
        else:
            realtime_mode()
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"\n系统级错误: {str(e)}")
        traceback.print_exc()
    finally:
        print("系统已关闭")