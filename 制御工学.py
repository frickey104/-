import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import control
import warnings

matplotlib.rcParams['font.family'] = 'Yu Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 14

class VCMControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VCM 制御特性解析ツール")
        self.root.geometry("1800x1200")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.create_widgets()
        self.plot_all()

    def create_widgets(self):
        font_label = ("Yu Gothic", 16)
        font_entry = ("Yu Gothic", 16)

        self.params = {}
        vcm_defaults = {"L (H)": 0.004, "R (Ω)": 20.0, "Kf (N/A)": 5.0, "m (kg)": 0.065, "c (Ns/m)": 0.05, "k (N/m)": 0.0}
        pid_defaults = {"Kp": 1000.0, "Ki": 0.0, "Kd": 20.0}
        sine_defaults = {"正弦波周波数 [rad/s]": 500.0}

        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        param_frame = ttk.LabelFrame(left_frame, text="パラメータ設定", padding=10)
        param_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        row = 0
        for label, default in {**vcm_defaults, **pid_defaults, **sine_defaults}.items():
            ttk.Label(param_frame, text=label, font=font_label).grid(row=row, column=0, sticky=tk.W, pady=5)
            val = tk.DoubleVar(value=default)
            entry = tk.Entry(param_frame, textvariable=val, width=10, font=font_entry)
            entry.grid(row=row, column=1, pady=5)
            self.params[label] = val
            row += 1

        ttk.Button(param_frame, text="グラフ更新", command=self.plot_all).grid(row=row, column=0, columnspan=2, pady=10)

        info_frame = ttk.LabelFrame(left_frame, text="解析結果", padding=10)
        info_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        self.gm_open_label = ttk.Label(info_frame, text="開ループ GM: --- dB", font=font_label)
        self.pm_open_label = ttk.Label(info_frame, text="開ループ PM: --- °", font=font_label)
        self.gm_closed_label = ttk.Label(info_frame, text="閉ループ GM: --- dB", font=font_label)
        self.pm_closed_label = ttk.Label(info_frame, text="閉ループ PM: --- °", font=font_label)
        self.info_label = ttk.Label(info_frame, text="", font=font_label, justify=tk.LEFT)

        self.gm_open_label.pack(anchor=tk.W, pady=2)
        self.pm_open_label.pack(anchor=tk.W, pady=2)
        self.gm_closed_label.pack(anchor=tk.W, pady=2)
        self.pm_closed_label.pack(anchor=tk.W, pady=2)
        self.info_label.pack(anchor=tk.W, pady=5)

        graph_frame = ttk.Frame(self.root)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(16, 14))
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 2, 2])
        self.ax_block = self.fig.add_subplot(gs[0, :])
        self.ax_mag = self.fig.add_subplot(gs[1, 0])
        self.ax_phase = self.fig.add_subplot(gs[1, 1])
        self.ax_step = self.fig.add_subplot(gs[2, 0])
        self.ax_sine = self.fig.add_subplot(gs[2, 1])
        self.fig.tight_layout(pad=4.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def plot_block_diagram(self):
        ax = self.ax_block
        ax.clear()
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)

        ax.text(0.5, 1.5, "目標値\n$r$ [m]", ha='center')
        ax.annotate("", xy=(1.2, 1.5), xytext=(0.8, 1.5), arrowprops=dict(arrowstyle="->", lw=2))
        ax.text(1.0, 1.6, "誤差", ha='center')

        ax.add_patch(plt.Rectangle((1.2, 1.3), 1.2, 0.4, fill=False))
        ax.text(1.8, 1.5, "PID\n$K_p$, $K_i$, $K_d$", ha='center', va='center')
        ax.annotate("", xy=(2.6, 1.5), xytext=(2.4, 1.5), arrowprops=dict(arrowstyle="->", lw=2))
        ax.text(2.5, 1.3, "$V$ [V]", ha='center')

        ax.add_patch(plt.Rectangle((2.6, 1.3), 1.2, 0.4, fill=False))
        ax.text(3.2, 1.5, "電気回路\n$L$, $R$", ha='center')
        ax.annotate("", xy=(4.0, 1.5), xytext=(3.8, 1.5), arrowprops=dict(arrowstyle="->", lw=2))
        ax.text(3.9, 1.3, "$i$ [A]", ha='center')

        ax.add_patch(plt.Rectangle((4.0, 1.3), 1.2, 0.4, fill=False))
        ax.text(4.6, 1.5, "力変換\n$K_f$", ha='center')
        ax.annotate("", xy=(5.4, 1.5), xytext=(5.2, 1.5), arrowprops=dict(arrowstyle="->", lw=2))
        ax.text(5.3, 1.3, "$F$ [N]", ha='center')

        ax.add_patch(plt.Rectangle((5.4, 1.3), 1.4, 0.4, fill=False))
        ax.text(6.1, 1.5, "機械系\n$m$, $c$, $k$", ha='center')
        ax.annotate("", xy=(6.8, 1.5), xytext=(6.6, 1.5), arrowprops=dict(arrowstyle="->", lw=2))
        ax.text(6.9, 2.0, "出力\n$x$ [m]", ha='left')

        ax.annotate("", xy=(0.8, 2.5), xytext=(6.8, 2.5), arrowprops=dict(arrowstyle='->', lw=1.5, linestyle='dashed'))
        ax.annotate("", xy=(0.8, 1.5), xytext=(0.8, 2.5), arrowprops=dict(arrowstyle='->', lw=1.5, linestyle='dashed'))
        ax.annotate("", xy=(6.8, 2.5), xytext=(6.8, 1.5), arrowprops=dict(arrowstyle='->', lw=1.5, linestyle='dashed'))
        ax.text(3.8, 2.6, "位置フィードバック", ha='center')

    def plot_all(self):
        self.plot_block_diagram()
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        p = self.params
        L, R, Kf = p["L (H)"].get(), p["R (Ω)"].get(), p["Kf (N/A)"].get()
        m, c, k = p["m (kg)"].get(), p["c (Ns/m)"].get(), p["k (N/m)"].get()
        Kp, Ki, Kd = p["Kp"].get(), p["Ki"].get(), p["Kd"].get()
        freq_rad = p["正弦波周波数 [rad/s]"].get()

        G1 = control.tf([1], [L, R])
        G2 = control.tf([Kf], [1])
        G3 = control.tf([1], [m, c, k])
        plant = G3 * G2 * G1
        pid = control.tf([Kd, Kp, Ki], [1, 0])
        open_loop = pid * plant
        closed_loop = control.feedback(open_loop, sign=-1)

        omega = np.logspace(1, 5, 1000)
        mag_open, phase_open, _ = control.bode(open_loop, omega=omega, plot=False)
        mag_closed, phase_closed, _ = control.bode(closed_loop, omega=omega, plot=False)
        mag_elec, phase_elec, _ = control.bode(G1, omega=omega, plot=False)
        mag_mech, phase_mech, _ = control.bode(G3, omega=omega, plot=False)

        self.ax_mag.clear()
        self.ax_mag.semilogx(omega, 20 * np.log10(mag_open), 'r--', label="開ループ")
        self.ax_mag.semilogx(omega, 20 * np.log10(mag_closed), 'b', label="閉ループ")
        self.ax_mag.semilogx(omega, 20 * np.log10(mag_elec), 'g--', label="電気系")
        self.ax_mag.semilogx(omega, 20 * np.log10(mag_mech), 'm--', label="機械系")
        self.ax_mag.set_ylabel("ゲイン[dB]"); self.ax_mag.grid(); self.ax_mag.legend()
        self.ax_mag.set_title("Bode ゲイン特性")

        self.ax_phase.clear()
        self.ax_phase.semilogx(omega, np.degrees(phase_open), 'r--', label="開ループ")
        self.ax_phase.semilogx(omega, np.degrees(phase_closed), 'b', label="閉ループ")
        self.ax_phase.semilogx(omega, np.degrees(phase_elec), 'g--', label="電気系")
        self.ax_phase.semilogx(omega, np.degrees(phase_mech), 'm--', label="機械系")
        self.ax_phase.set_ylabel("位相[deg]"); self.ax_phase.set_xlabel("周波数[rad/s]")
        self.ax_phase.grid(); self.ax_phase.legend()
        self.ax_phase.set_title("Bode 位相特性")

        t_step, y_step = control.step_response(closed_loop)
        self.ax_step.clear()
        self.ax_step.plot(t_step, np.ones_like(t_step), 'k--', label="入力（ステップ）")
        self.ax_step.plot(t_step, y_step, 'b', label="閉ループ")
        self.ax_step.set_title("ステップ応答"); self.ax_step.grid(); self.ax_step.legend()

        overshoot = (np.max(y_step) - 1) * 100
        rise_time = t_step[np.where(y_step >= 1)[0][0]] if np.any(y_step >= 1) else np.nan

        eps = 0.02
        out_of_range = np.where(np.abs(y_step - 1.0) > eps)[0]
        if len(out_of_range) == 0:
            settling_time = t_step[-1]
        else:
            last_out = out_of_range[-1]
            settling_time = t_step[last_out + 1] if last_out + 1 < len(t_step) else np.nan

        # ★ 追加部分：整定時間の2倍まで表示
        self.ax_step.set_xlim(0, settling_time * 2)

        steady_error = np.abs(1 - y_step[-1])
        info = f"【性能指標】\n立上り時間: {rise_time:.4f} s\nオーバーシュート: {overshoot:.2f} %\n整定時間: {settling_time:.4f} s\n定常偏差: {steady_error:.4f}"
        self.info_label.config(text=info)

        t_sin = np.linspace(0, 0.05, 1000)
        u_sin = np.sin(freq_rad * t_sin)
        t_out, y_cl = control.forced_response(closed_loop, T=t_sin, U=u_sin)
        self.ax_sine.clear()
        self.ax_sine.plot(t_out, u_sin, 'k--', label="入力（sin波）")
        self.ax_sine.plot(t_out, y_cl, 'b', label="閉ループ")
        self.ax_sine.set_title("正弦波応答"); self.ax_sine.grid(); self.ax_sine.legend()

        gm_open, pm_open, _, _ = control.margin(open_loop)
        gm_closed, pm_closed, _, _ = control.margin(closed_loop)
        gm_db_open = 20 * np.log10(gm_open) if gm_open and np.isfinite(gm_open) else np.inf
        gm_db_closed = 20 * np.log10(gm_closed) if gm_closed and np.isfinite(gm_closed) else np.inf

        self.gm_open_label.config(text=f"開ループ GM: {gm_db_open:.2f} dB" if np.isfinite(gm_db_open) else "開ループ GM: ∞ dB")
        self.pm_open_label.config(text=f"開ループ PM: {pm_open:.2f} °" if np.isfinite(pm_open) else "開ループ PM: ∞ °")
        self.gm_closed_label.config(text=f"閉ループ GM: {gm_db_closed:.2f} dB" if np.isfinite(gm_db_closed) else "閉ループ GM: ∞ dB")
        self.pm_closed_label.config(text=f"閉ループ PM: {pm_closed:.2f} °" if np.isfinite(pm_closed) else "閉ループ PM: ∞ °")

        self.canvas.draw()

    def on_close(self):
        plt.close(self.fig)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VCMControlApp(root)
    root.mainloop()
