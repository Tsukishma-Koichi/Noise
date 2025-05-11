# main.py
import os
import time
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import librosa
import soundfile as sf
import matplotlib
import queue
from dotenv import load_dotenv
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from realtime_processor import RealTimeProcessor
from analysis import analyze_audio
from classifier import NoiseClassifier
from filters import AudioFilter
from visualization import AudioVisualizer
import matplotlib.pyplot as plt

load_dotenv()

class NoiseReductionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self._init_font_config()
        self.title("智能降噪系统 V8.5")
        self.geometry("1200x800")
        self.processing = False
        self.current_plots = {}
        self.realtime_window = None
        self._setup_ui()
        self.protocol("WM_DELETE_WINDOW", self._safe_exit)

    def _init_font_config(self):
        try:
            font_spec = ('Microsoft YaHei', 10)
            self.option_add('*Font', font_spec)
            test_label = ttk.Label(self, text="测试字体", font=font_spec)
            test_label.destroy()
        except tk.TclError:
            self.option_add('*Font', ('TkDefaultFont', 10))

    def _setup_ui(self):
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#F0F0F0')
        self.style.configure('Title.TLabel',
                            font=('Microsoft YaHei', 16, 'bold'),
                            background='#4B8BBE',
                            foreground='white')

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 标题栏
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(title_frame, text="智能降噪系统", style='Title.TLabel').pack(expand=True, fill=tk.X)

        # 控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        ttk.Button(control_frame, text="文件降噪", command=self._start_file_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="实时模式", command=self._start_realtime).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="退出系统", command=self._safe_exit).pack(side=tk.RIGHT, padx=5)

        # 可视化区域
        self.visual_paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, sashwidth=8)
        self.visual_paned.pack(fill=tk.BOTH, expand=True)

        # 波形对比区域
        self.wave_frame = ttk.Frame(self.visual_paned)
        self._init_plot_area(self.wave_frame, "波形对比")
        self.visual_paned.add(self.wave_frame, minsize=600)

        # 频谱分析区域
        self.spec_frame = ttk.Frame(self.visual_paned)
        self._init_plot_area(self.spec_frame, "频谱分析")
        self.visual_paned.add(self.spec_frame, minsize=600)

        # 状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        self.progress = ttk.Progressbar(status_frame, mode='determinate')
        self.progress.pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(status_frame, text="就绪")
        self.status_label.pack(side=tk.LEFT)

        # 日志控制台
        console_frame = ttk.LabelFrame(main_frame, text="处理日志")
        console_frame.pack(fill=tk.BOTH, expand=False, pady=5)
        self.console = scrolledtext.ScrolledText(console_frame,
                                                wrap=tk.WORD,
                                                state='disabled',
                                                font=('Consolas', 9),
                                                height=8)
        self.console.pack(fill=tk.BOTH, expand=True)

    def _init_plot_area(self, parent, title):
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X)
        ttk.Label(title_frame, text=title, font=('Microsoft YaHei', 10)).pack(side=tk.LEFT)
        plot_container = ttk.Frame(parent)
        plot_container.pack(fill=tk.BOTH, expand=True)
        self.current_plots[title] = {'figure': None, 'canvas': None, 'toolbar': None}

    def _log_message(self, message, level="INFO"):
        tag_colors = {"INFO": "black", "WARNING": "orange", "ERROR": "red", "SUCCESS": "green"}
        self.console.configure(state='normal')
        self.console.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n", level)
        self.console.tag_config(level, foreground=tag_colors.get(level, "black"))
        self.console.configure(state='disabled')
        self.console.see(tk.END)

    def _start_file_processing(self):
        if self.processing:
            messagebox.showwarning("警告", "当前有任务正在处理中")
            return
        file_path = filedialog.askopenfilename(filetypes=[("音频文件", "*.wav *.mp3 *.ogg")])
        if not file_path: return
        self._set_ui_state(False)
        threading.Thread(target=self._process_audio_file, args=(file_path,), daemon=True).start()

    def _process_audio_file(self, file_path):
        try:
            self._update_progress(10, "加载音频文件中...")
            y, sr = librosa.load(file_path, sr=None, mono=True)

            self._update_progress(20, "执行噪声分析...")
            features = analyze_audio(file_path)
            self._log_message(f"噪声特征: {features}", "INFO")

            classifier = NoiseClassifier()
            result = classifier.classify(features)
            self._log_message(f"分类结果: {result['classification']}", "INFO")

            self._update_progress(40, "执行降噪处理...")
            if result['classification'] == "steady":
                processed = AudioFilter.iir_notch_filter(y, sr)
            else:
                processed = AudioFilter.lms_adaptive_filter(y, sr)

            self._update_progress(70, "保存处理结果...")
            output_path = self._save_processed_audio(processed, sr, file_path)

            self._update_progress(90, "生成可视化...")
            self.after(0, self._show_advanced_results, y, processed, sr, file_path)

            self._log_message(f"处理完成: {output_path}", "SUCCESS")

        except Exception as e:
            self._log_message(f"处理失败: {str(e)}", "ERROR")
            traceback.print_exc()
            self.after(0, messagebox.showerror, "错误", str(e))
        finally:
            self._set_ui_state(True)
            self._update_progress(100, "就绪")

    def _save_processed_audio(self, signal, sr, original_path):
        output_dir = "processed"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_path = os.path.join(output_dir, f"clean_{base_name}.wav")
        sf.write(output_path, signal, sr)
        return output_path

    def _show_advanced_results(self, original, processed, sr, file_path):
        # 基础波形对比
        self._update_plot(
            self.wave_frame,
            original,
            processed,
            sr,
            "时间 (秒)",
            "振幅",
            "波形对比"
        )

        # 频谱分析
        self._update_plot(
            self.spec_frame,
            processed,
            None,
            sr,
            "频率 (Hz)",
            "能量 (dB)",
            "频谱分析",
            specgram=True
        )

        # 高级分析视图
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        fig = Figure(figsize=(12, 8), dpi=100)
        ax1 = fig.add_subplot(2, 1, 1)
        AudioVisualizer.plot_wave_comparison(original, processed, sr, ax1)
        ax1.set_title("波形对比 - " + base_name)
        ax2 = fig.add_subplot(2, 1, 2)
        AudioVisualizer.plot_spectrogram(processed, sr, ax2)
        ax2.set_title("降噪后语谱图")
        fig.tight_layout()

        new_window = tk.Toplevel(self)
        new_window.title("高级分析视图")
        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, new_window)
        toolbar.update()

    def _update_plot(self, parent, data1, data2, sr, xlabel, ylabel, title, specgram=False):
        if title in self.current_plots:
            self._clear_plot(title)

        figure = Figure(figsize=(8, 4), dpi=100)
        ax = figure.add_subplot(111)

        if specgram:
            S = librosa.amplitude_to_db(np.abs(librosa.stft(data1)), ref=np.max)
            img = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', ax=ax)
            figure.colorbar(img, ax=ax, format='%+2.0f dB')
        else:
            t = np.arange(len(data1)) / sr
            line1, = ax.plot(t, data1, alpha=0.7, label='原始信号')
            if data2 is not None:
                line2, = ax.plot(t, data2, alpha=0.9, label='降噪信号')
            ax.legend(loc='upper right')

        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, facecolor='#F0F0F0')
        ax.grid(True, linestyle='--', alpha=0.6)
        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        self.current_plots[title] = {
            'figure': figure,
            'canvas': canvas,
            'toolbar': toolbar
        }
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        figure.canvas.mpl_connect('close_event', lambda e: self._on_plot_close(title))

    def _clear_plot(self, title):
        if self.current_plots[title]['canvas']:
            self.current_plots[title]['canvas'].get_tk_widget().destroy()
            self.current_plots[title]['toolbar'].destroy()
            self.current_plots[title] = {'figure': None, 'canvas': None, 'toolbar': None}

    def _on_plot_close(self, title):
        self.current_plots[title]['figure'].clf()
        plt.close(self.current_plots[title]['figure'])
        self._clear_plot(title)

    def _start_realtime(self):
        if self.realtime_window is None or not self.realtime_window.winfo_exists():
            self.realtime_window = RealtimeWindow(self)
            self.realtime_window.grab_set()

    def _set_ui_state(self, enabled):
        state = 'normal' if enabled else 'disabled'
        self.processing = not enabled
        for child in self.winfo_children():
            if isinstance(child, ttk.Button):
                child.configure(state=state)

    def _update_progress(self, value, message):
        self.progress['value'] = value
        self.status_label['text'] = message

    def _safe_exit(self):
        try:
            if self.processing and not messagebox.askokcancel("退出确认", "当前有任务正在运行，确定要强制退出吗？"):
                return
            for child in self.winfo_children():
                if isinstance(child, tk.Toplevel):
                    child.destroy()
            self._clean_tcl_commands()
            self.quit()
            self.destroy()
        except Exception as e:
            print(f"退出时发生错误: {str(e)}")
            os._exit(1)

    def _clean_tcl_commands(self):
        try:
            commands = self.tk.eval('info commands')
            for cmd in commands.split():
                if not cmd.startswith(('tk', 'ttk', '::tk', '::ttk')):
                    try: self.tk.eval(f'rename {cmd} {{}}')
                    except: pass
        except Exception as e:
            print(f"清理Tcl命令失败: {str(e)}")

class RealtimeWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("实时降噪监控")
        self.geometry("800x600")
        self.processor = RealTimeProcessor()
        self._setup_ui()
        self.protocol("WM_DELETE_WINDOW", self._safe_close)

    def _setup_ui(self):
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        self.start_btn = ttk.Button(btn_frame, text="启动", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.raw_line, = self.ax.plot([], [], 'r-', alpha=0.7, label='原始SNR')
        self.proc_line, = self.ax.plot([], [], 'g-', lw=1.5, label='降噪SNR')
        self.avg_line = self.ax.axhline(0, color='b', ls='--', label='平均SNR')
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(-5, 20)
        self.ax.set_xlabel('时间 (秒)')
        self.ax.set_ylabel('SNR (dB)')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()

    def start(self):
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.processor.start(self._update_plot)
        self.after(100, self._check_queue)

    def _check_queue(self):
        if self.processor.running:
            self._update_plot()
            self.after(100, self._check_queue)

    def _update_plot(self):
        try:
            data = self.processor.plot_queue.get_nowait()
            raw_t, raw_s, proc_t, proc_s, avg = data
            self.raw_line.set_data(raw_t, raw_s)
            self.proc_line.set_data(proc_t, proc_s)
            self.avg_line.set_ydata([avg, avg])
            if raw_t:
                current_time = max(raw_t[-1], proc_t[-1])
                self.ax.set_xlim(max(0, current_time - 10), current_time + 1)
                min_snr = min(min(raw_s), min(proc_s)) - 2
                max_snr = max(max(raw_s), max(proc_s)) + 2
                self.ax.set_ylim(min_snr if min_snr > -10 else -10,
                                 max_snr if max_snr < 50 else 50)
            self.canvas.draw_idle()
        except queue.Empty:
            pass

    def stop(self):
        self.processor.stop()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _safe_close(self):
        try:
            self.processor.stop()
            for id_ in self.tk.eval('after info').split():
                self.after_cancel(id_)
            self.destroy()
        except Exception as e:
            print(f"关闭实时窗口失败: {str(e)}")

if __name__ == "__main__":
    app = NoiseReductionApp()
    app.mainloop()