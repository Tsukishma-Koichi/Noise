import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime

def record_audio():
    """实时音频采集（采样率16000Hz，单声道）"""
    fs = 16000
    channels = 1
    filename = f"recording_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"

    print("\n麦克风准备中...")
    try:
        recorded_data = []

        def audio_callback(indata, frames, time, status):
            recorded_data.append(indata.copy())

        with sd.InputStream(samplerate=fs, channels=channels, callback=audio_callback):
            print("录音开始，按下键盘P键结束采集...")
            while True:
                if keyboard.is_pressed('p'):
                    print("检测到停止指令，正在保存文件...")
                    break
                time.sleep(0.1)

        audio_array = np.concatenate(recorded_data, axis=0)
        sf.write(filename, audio_array, fs)
        print(f"音频已保存至：{os.path.abspath(filename)}")
        return filename

    except Exception as e:
        raise RuntimeError(f"录音失败: {str(e)}")