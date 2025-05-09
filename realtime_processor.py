# realtime_processor.py
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import keyboard
from scipy.signal import lfilter, iirnotch, lfilter_zi
from collections import deque
import matplotlib

matplotlib.use('TkAgg')
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

        # 图形初始化
        self._init_plot()
        self.plot_queue = queue.Queue(maxsize=5)
        self.ani = None

    def _init_fonts(self):
        """安全初始化中文字体"""
        try:
            font_path = os.path.abspath('fonts/simhei.ttf')
            if not os.path.exists(font_path):
                raise FileNotFoundError(f"字体文件未找到: {font_path}")

            self.font = FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = [self.font.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"字体初始化警告: {str(e)}")
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

    def _init_plot(self):
        """增强的图形初始化"""
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(10, 4))

        # 创建图形元素
        self.line, = self.ax.plot([], [], 'g-', lw=1, label='瞬时SNR')
        self.avg_line = self.ax.axhline(0, color='r', ls='--', label='平均SNR')

        # 设置中文标签
        self.ax.set_xlabel('时间 (秒)', fontproperties=self.font)
        self.ax.set_ylabel('SNR (dB)', fontproperties=self.font)

        # 配置图例
        self.ax.legend(prop=self.font, loc='upper right')

        # 初始坐标范围
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-10, 50)
        self.ax.grid(True, alpha=0.3)

        plt.ion()
        self.fig.show()  # 预显示窗口

    def _animate_update(self, frame):
        """动画帧更新函数"""
        try:
            if not self.plot_queue.empty():
                times, snrs, avg = self.plot_queue.get()

                # 更新曲线数据
                self.line.set_data(np.array(times) - times[-1] + 10, snrs)
                self.avg_line.set_ydata([avg] * 2)

                # 自动调整坐标范围
                if len(times) > 1:
                    self.ax.set_xlim(max(0, times[-1] - 10), max(10, times[-1]))
                    self.ax.figure.canvas.draw()
        except Exception as e:
            pass
        return self.line, self.avg_line

    def audio_callback(self, indata, frames, time, status):
        """音频输入回调"""
        if status:
            print(f"音频流异常: {status}")
        if self.running:
            try:
                self.audio_buffer.put_nowait(indata.copy())
            except queue.Full:
                pass

    def processing_loop(self):
        """实时处理主循环"""
        try:
            time_counter = 0
            last_update = time.time()

            while self.running:
                # 获取数据块
                try:
                    data = self.audio_buffer.get(timeout=0.5)
                except queue.Empty:
                    continue

                # 处理数据
                chunk = data.flatten()
                filtered = self._apply_filter(chunk)
                current_snr = self._calculate_snr(chunk, filtered)
                time_counter += len(chunk) / self.fs

                # 更新历史数据
                self.snr_history.append((time_counter, current_snr))

                # 定时更新显示队列
                if time.time() - last_update >= self.update_interval:
                    try:
                        self.plot_queue.put_nowait((
                            [t for t, _ in self.snr_history],
                            [s for _, s in self.snr_history],
                            np.mean([s for _, s in self.snr_history]) if self.snr_history else 0
                        ))
                    except queue.Full:
                        pass
                    last_update = time.time()

        except Exception as e:
            print(f"处理线程异常: {str(e)}")
            self.stop()

    # 保持滤波和信号处理方法不变（_apply_filter, _notch_filter等）

    def _apply_filter(self, chunk):
        """应用当前滤波器（新增关键方法）"""
        if self.filter_type == "steady":
            return self._notch_filter(chunk)
        else:
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
        """实时SNR计算方法（新增关键方法）"""
        noise = original - processed
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)

        # 数值稳定性处理
        eps = 1e-12
        return 10 * np.log10((signal_power + eps) / (noise_power + eps))

    def start(self):
        """启动实时处理系统"""
        if self.running:
            return

        self.running = True

        # 启动动画
        self.ani = FuncAnimation(
            self.fig,
            self._animate_update,
            interval=50,
            blit=False,
            cache_frame_data=False
        )

        # 启动音频处理线程
        self.audio_thread = threading.Thread(
            target=self._audio_stream_loop,
            daemon=True
        )
        self.audio_thread.start()

        # 启动处理线程
        self.process_thread = threading.Thread(
            target=self.processing_loop,
            daemon=True
        )
        self.process_thread.start()

        # 显示图形界面
        plt.show(block=True)  # 阻塞主线程直到窗口关闭

    def _audio_stream_loop(self):
        """音频流专用线程"""
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
                    time.sleep(0.01)
        except Exception as e:
            self.stop()

    def stop(self):
        """增强的资源释放方法"""
        if not self.running:
            return

        self.running = False

        # 停止动画
        if self.ani:
            self.ani.event_source.stop()

        # 关闭图形窗口
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)

        # 等待线程结束
        threads = []
        for t in [self.audio_thread, self.process_thread]:
            if t and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=1)
                threads.append(t)

        print(f"已停止 {len(threads)} 个工作线程")
        print("系统已完全停止")