# realtime_processor.py
import threading
import queue
import time
import numpy as np
import sounddevice as sd
from scipy.signal import lfilter, iirnotch, lfilter_zi
from collections import deque
import matplotlib
import keyboard

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties
import os


class RealTimeProcessor:
    def __init__(self, fs=16000, block_size=1024, update_interval=1.0):
        # 初始化字体配置
        self._init_fonts()

        # 音频参数
        self.fs = fs
        self.block_size = block_size
        self.update_interval = update_interval

        # 处理控制
        self.running = False
        self.audio_buffer = queue.Queue(maxsize=20)
        self.snr_history = deque(maxlen=int(10 / update_interval))

        # 滤波器参数
        self.filter_type = "non-steady"
        self.notch_params = {"Q": 30, "zi": None, "b": None, "a": None}
        self.lms_params = {
            "order": 128,
            "mu": 0.1,
            "weights": np.zeros(128),
            "reference": None
        }

        # 图形系统初始化
        self._init_plot()

    def _init_fonts(self):
        """安全初始化中文字体"""
        try:
            font_path = os.path.abspath('fonts/simhei.ttf')
            if os.path.exists(font_path):
                self.font = FontProperties(fname=font_path)
            else:
                self.font = FontProperties(family='SimHei')
        except Exception as e:
            print(f"字体初始化警告: {str(e)}")
            self.font = FontProperties(family='SimHei')

        plt.rcParams['font.sans-serif'] = [self.font.get_name()]
        plt.rcParams['axes.unicode_minus'] = False

    def _init_plot(self):
        """图形系统初始化"""
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(10, 4))

        # 初始化曲线
        self.line, = self.ax.plot([], [], 'g-', lw=1, label='瞬时SNR')
        self.avg_line = self.ax.axhline(0, color='r', ls='--', label='平均SNR')

        # 坐标轴配置
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-5, 5)
        self.ax.set_xlabel('时间 (秒)', fontproperties=self.font)
        self.ax.set_ylabel('SNR (dB)', fontproperties=self.font)
        self.ax.legend(prop=self.font)
        self.ax.grid(True, alpha=0.3)

        # 初始化动画
        self.ani = FuncAnimation(
            self.fig,
            self._animate_update,
            interval=50,
            blit=False,
            cache_frame_data=False
        )

        plt.ion()
        self.plot_queue = queue.Queue(maxsize=5)

    def _animate_update(self, frame):
        """动画更新函数"""
        try:
            times, snrs, avg = self.plot_queue.get_nowait()

            # 更新曲线数据
            self.line.set_data(times, snrs)
            self.avg_line.set_ydata([avg, avg])

            # 动态调整坐标
            if len(times) > 0:
                current_time = times[-1]
                self.ax.set_xlim(max(0, current_time - 10), current_time + 0.1)
                min_snr = min(snrs) - 2 if len(snrs) > 0 else -5
                max_snr = max(snrs) + 2 if len(snrs) > 0 else 5
                self.ax.set_ylim(min_snr, max_snr)

            self.fig.canvas.draw_idle()
            return self.line, self.avg_line
        except queue.Empty:
            return self.line, self.avg_line

    def audio_callback(self, indata, frames, time, status):
        """音频输入回调"""
        if status:
            print(f"音频流异常: {status}")
        if self.running and not self.audio_buffer.full():
            self.audio_buffer.put(indata.copy())

    def processing_loop(self):
        """实时处理主循环"""
        time_counter = 0.0
        last_update = time.time()

        while self.running:
            try:
                # 获取音频数据
                data = self.audio_buffer.get(timeout=0.5)
                chunk = data.flatten()

                # 处理数据
                filtered = self._apply_filter(chunk)
                current_snr = self._calculate_snr(chunk, filtered)
                time_counter += len(chunk) / self.fs

                # 记录数据
                self.snr_history.append((time_counter, current_snr))

                # 更新图形队列
                if time.time() - last_update >= self.update_interval:
                    times = [t for t, _ in self.snr_history]
                    snrs = [s for _, s in self.snr_history]
                    avg = np.mean(snrs) if snrs else 0

                    if not self.plot_queue.full():
                        self.plot_queue.put((times.copy(), snrs.copy(), avg))
                    last_update = time.time()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理异常: {str(e)}")
                self.stop()

    def _apply_filter(self, chunk):
        """应用当前滤波器"""
        if self.filter_type == "steady":
            return self._notch_filter(chunk)
        return self._lms_filter(chunk)

    def _notch_filter(self, chunk):
        """陷波滤波器实现"""
        if self.notch_params["zi"] is None or time.time() % 2 < 0.1:
            main_freq = self._find_main_frequency(chunk)
            w0 = main_freq / (self.fs / 2)
            self.notch_params["b"], self.notch_params["a"] = iirnotch(w0, self.notch_params["Q"])
            self.notch_params["zi"] = lfilter_zi(self.notch_params["b"], self.notch_params["a"]) * chunk[0]

        filtered, self.notch_params["zi"] = lfilter(
            self.notch_params["b"],
            self.notch_params["a"],
            chunk,
            zi=self.notch_params["zi"]
        )
        return filtered

    def _lms_filter(self, chunk):
        """LMS自适应滤波器实现"""
        if self.lms_params["reference"] is None:
            self.lms_params["reference"] = np.random.randn(len(chunk)) * 0.1

        filtered = np.zeros_like(chunk)
        for i in range(len(chunk)):
            if i < self.lms_params["order"]:
                filtered[i] = chunk[i]
                continue
            x = self.lms_params["reference"][i - self.lms_params["order"]:i]
            filtered[i] = np.dot(self.lms_params["weights"], x)
            e = chunk[i] - filtered[i]
            self.lms_params["weights"] += self.lms_params["mu"] * e * x
        return filtered

    def _find_main_frequency(self, chunk):
        """主频检测方法"""
        fft = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1 / self.fs)
        return freqs[np.argmax(fft)]

    def _calculate_snr(self, original, processed):
        """SNR计算方法"""
        noise = original - processed
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        eps = 1e-12
        return 10 * np.log10((signal_power + eps) / (noise_power + eps))

    def start(self):
        """启动系统"""
        if self.running:
            return

        self.running = True

        # 启动处理线程
        self.process_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.process_thread.start()

        # 启动音频流
        self.audio_thread = threading.Thread(target=self._audio_stream_loop, daemon=True)
        self.audio_thread.start()

        # 显示图形界面
        plt.show(block=True)

    def _audio_stream_loop(self):
        """音频流线程"""
        try:
            with sd.InputStream(
                    samplerate=self.fs,
                    blocksize=self.block_size,
                    channels=1,
                    callback=self.audio_callback
            ):
                print("实时降噪运行中，按 Q 键停止...")
                while self.running:
                    if keyboard.is_pressed('q'):
                        self.stop()
                    time.sleep(0.1)
        except Exception as e:
            self.stop()

    def stop(self):
        """停止系统"""
        if not self.running:
            return

        self.running = False

        # 关闭图形
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)

        # 等待线程结束
        for t in [self.process_thread, self.audio_thread]:
            if t and t.is_alive():
                t.join(timeout=1)

        print("系统已安全停止")