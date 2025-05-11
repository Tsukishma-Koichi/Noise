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

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']  # 设置中文字体列表
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class RealTimeProcessor:
    def __init__(self, fs=16000, block_size=1024, update_interval=1.0):
        # 音频参数
        self.fs = fs
        self.block_size = block_size
        self.update_interval = update_interval

        # 处理控制
        self.running = False
        self.audio_buffer = queue.Queue(maxsize=20)
        self.raw_snr_history = deque(maxlen=int(10 / update_interval))
        self.proc_snr_history = deque(maxlen=int(10 / update_interval))

        # 滤波器参数
        self.filter_type = "non-steady"
        self.notch_params = {"Q": 30, "zi": None, "b": None, "a": None}
        self.lms_params = {
            "order": 128,
            "mu": 0.1,
            "weights": np.zeros(128),
            "reference": None
        }

        # 数据通信
        self.plot_queue = queue.Queue(maxsize=8)

    def _init_fonts(self):
        """安全初始化中文字体"""
        try:
            # 尝试使用系统自带字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
            plt.rcParams['axes.unicode_minus'] = False

            # 验证字体是否生效
            test_font = plt.get_cachedir() + '/font_test.png'
            plt.figure()
            plt.text(0.5, 0.5, '测试中文字体', ha='center')
            plt.savefig(test_font, dpi=50)
            plt.close()
        except Exception as e:
            print(f"字体初始化警告: {str(e)}")
            # 回退到英文显示
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

    def _init_plot(self):
        self._init_fonts()  # 添加字体初始化
        """初始化双曲线图形界面"""
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))

        # 初始化两种曲线
        self.raw_line, = self.ax.plot([], [], 'r-', alpha=0.7, label='原始SNR')
        self.proc_line, = self.ax.plot([], [], 'g-', lw=1.5, label='降噪SNR')
        self.avg_line = self.ax.axhline(0, color='b', ls='--', label='平均SNR')

        # 图形配置
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-5, 20)
        # 修改所有中文标签为英文（避免警告）
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('SNR (dB)')
        self.ax.legend(['Raw SNR', 'Processed SNR', 'Average'], loc='upper right')
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
        self.plot_queue = queue.Queue(maxsize=8)

    def _animate_update(self, frame):
        """动画更新函数"""
        try:
            data = self.plot_queue.get_nowait()
            raw_t, raw_snr, proc_t, proc_snr, avg = data

            # 更新原始SNR曲线
            self.raw_line.set_data(raw_t, raw_snr)

            # 更新降噪SNR曲线
            self.proc_line.set_data(proc_t, proc_snr)

            # 更新平均线
            self.avg_line.set_ydata([avg, avg])

            # 动态调整坐标
            if len(raw_t) > 0:
                current_time = max(max(raw_t), max(proc_t))
                self.ax.set_xlim(max(0, current_time - 10), current_time + 1)

                min_snr = min(min(raw_snr), min(proc_snr)) - 2
                max_snr = max(max(raw_snr), max(proc_snr)) + 2
                self.ax.set_ylim(min_snr if min_snr > -10 else -10,
                                 max_snr if max_snr < 50 else 50)

            self.fig.canvas.draw_idle()
        except (queue.Empty, ValueError):
            pass
        return self.raw_line, self.proc_line, self.avg_line

    def audio_callback(self, indata, frames, time, status):
        """音频输入回调（线程安全）"""
        if status:
            print(f"音频输入异常: {status}")
        if self.running and not self.audio_buffer.full():
            self.audio_buffer.put(indata.copy())

    def processing_loop(self, update_callback):
        """主处理循环"""
        time_counter = 0.0
        last_update = time.time()

        while self.running:
            try:
                # 获取音频数据
                data = self.audio_buffer.get(timeout=0.5)
                raw_chunk = data.flatten()

                # 信号处理
                processed = self._apply_filter(raw_chunk)

                # SNR计算
                raw_snr = self._calculate_raw_snr(raw_chunk)
                proc_snr = self._calculate_processed_snr(raw_chunk, processed)
                time_counter += len(raw_chunk) / self.fs

                # 记录历史数据
                self.raw_snr_history.append((time_counter, raw_snr))
                self.proc_snr_history.append((time_counter, proc_snr))

                # 定期触发界面更新
                if time.time() - last_update >= self.update_interval:
                    raw_t = [t for t, _ in self.raw_snr_history]
                    raw_s = [s for _, s in self.raw_snr_history]
                    proc_t = [t for t, _ in self.proc_snr_history]
                    proc_s = [s for _, s in self.proc_snr_history]
                    avg = np.mean(proc_s) if proc_s else 0

                    if not self.plot_queue.full():
                        self.plot_queue.put((
                            raw_t.copy(),
                            raw_s.copy(),
                            proc_t.copy(),
                            proc_s.copy(),
                            avg
                        ))
                    last_update = time.time()
                    update_callback()  # 触发主线程更新

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
            self.notch_params["b"], self.notch_params["a"] = iirnotch(w0, 30)
            self.notch_params["zi"] = lfilter_zi(
                self.notch_params["b"],
                self.notch_params["a"]
            ) * chunk[0]

        filtered, self.notch_params["zi"] = lfilter(
            self.notch_params["b"],
            self.notch_params["a"],
            chunk,
            zi=self.notch_params["zi"]
        )
        return filtered

    def _lms_filter(self, chunk):
        """LMS自适应滤波器"""
        if self.lms_params["reference"] is None:
            self.lms_params["reference"] = np.random.randn(len(chunk)) * 0.01

        filtered = np.zeros_like(chunk)
        for i in range(len(chunk)):
            if i < self.lms_params["order"]:
                filtered[i] = chunk[i]
                continue
            x = self.lms_params["reference"][i - self.lms_params["order"]:i]
            filtered[i] = np.dot(self.lms_params["weights"], x)
            e = chunk[i] - filtered[i]
            self.lms_params["weights"] += 0.1 * e * x
        return filtered

    def _find_main_frequency(self, chunk):
        """主频检测"""
        fft = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1 / self.fs)
        return freqs[np.argmax(fft)]

    def _calculate_raw_snr(self, chunk):
        """原始信号SNR估算"""
        noise_est = chunk[:len(chunk) // 4]
        signal_part = chunk[len(chunk) // 4:]
        sig_power = np.mean(signal_part ** 2)
        noise_power = np.mean(noise_est ** 2)
        return 10 * np.log10((sig_power + 1e-12) / (noise_power + 1e-12))

    def _calculate_processed_snr(self, original, processed):
        """降噪后SNR计算"""
        residual = original - processed
        sig_power = np.mean(processed ** 2)
        noise_power = np.mean(residual ** 2)
        return 10 * np.log10((sig_power + 1e-12) / (noise_power + 1e-12))

    def start(self, update_callback):
        """启动处理系统"""
        if self.running:
            return

        self.running = True

        # 启动处理线程
        self.process_thread = threading.Thread(
            target=self.processing_loop,
            args=(update_callback,),
            daemon=True
        )
        self.process_thread.start()

        # 启动音频采集线程
        self.audio_thread = threading.Thread(
            target=self._audio_stream_loop,
            daemon=True
        )
        self.audio_thread.start()

    def _audio_stream_loop(self):
        """音频流线程"""
        try:
            with sd.InputStream(
                    samplerate=self.fs,
                    blocksize=self.block_size,
                    channels=1,
                    callback=self.audio_callback
            ):
                print("实时降噪已启动 | 按 Q 键停止")
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            self.stop()

    def stop(self):
        """安全停止系统"""
        if not self.running:
            return

        self.running = False
        for t in [self.audio_thread, self.process_thread]:
            if t and t.is_alive():
                t.join(timeout=1)
        print("系统已安全关闭")

