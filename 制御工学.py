import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import control

matplotlib.rcParams['font.family'] = 'Yu Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

class VCMControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VCM 開ループ & 閉ループ ボード線図")
        self.root.geometry("1800x1000")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.create_widgets()
        self.plot_bode()

    def create_widgets(self):
        font_big = ("Yu Gothic", 14)

        param_frame = ttk.LabelFrame(self.root, text="VCM・PID パラメータ設定")
        param_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

        self.params = {}
        vcm_defaults = {
            "L (H)": 0.005,
            "R (Ω)": 2.0,
            "Kf (N/A)": 1.5,
            "m (kg)": 0.01,
            "c (Ns/m)": 0.001,
            "k (N/m)": 0.0
        }
        pid_defaults = {
            "Kp": 5.0,
            "Ki": 100.0,
            "Kd": 0.001
        }

        style = ttk.Style()
        style.configure("TLabel", font=font_big)
        style.configure("TEntry", font=font_big)
        style.configure("TButton", font=font_big)

        row = 0
        for label, default in vcm_defaults.items():
            ttk.Label(param_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
            val = tk.DoubleVar(value=default)
            entry = ttk.Entry(param_frame, textvariable=val, width=10)
            entry.grid(row=row, column=1, pady=5)
            self.params[label] = val
            row += 1

        ttk.Separator(param_frame).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1

        for label, default in pid_defaults.items():
            ttk.Label(param_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
            val = tk.DoubleVar(value=default)
            entry = ttk.Entry(param_frame, textvariable=val, width=10)
            entry.grid(row=row, column=1, pady=5)
            self.params[label] = val
            row += 1

        ttk.Button(param_frame, text="グラフ更新", command=self.plot_bode).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1

        self.gm_open_label = ttk.Label(param_frame, text="開ループ GM: --- dB")
        self.pm_open_label = ttk.Label(param_frame, text="開ループ PM: --- °")
        self.gm_open_label.grid(row=row, column=0, columnspan=2, pady=5)
        self.pm_open_label.grid(row=row+1, column=0, columnspan=2, pady=5)
        row += 2

        self.gm_closed_label = ttk.Label(param_frame, text="閉ループ GM: --- dB")
        self.pm_closed_label = ttk.Label(param_frame, text="閉ループ PM: --- °")
        self.gm_closed_label.grid(row=row, column=0, columnspan=2, pady=5)
        self.pm_closed_label.grid(row=row+1, column=0, columnspan=2, pady=5)

        graph_frame = ttk.Frame(self.root)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, (self.ax_mag, self.ax_phase) = plt.subplots(2, 1, figsize=(12, 9))
        self.fig.tight_layout(pad=5.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def plot_bode(self):
        p = self.params
        L, R, Kf = p["L (H)"].get(), p["R (Ω)"].get(), p["Kf (N/A)"].get()
        m, c, k = p["m (kg)"].get(), p["c (Ns/m)"].get(), p["k (N/m)"].get()
        Kp, Ki, Kd = p["Kp"].get(), p["Ki"].get(), p["Kd"].get()

        G1 = control.tf([1], [L, R])
        G2 = control.tf([Kf], [1])
        G3 = control.tf([1], [m, c, k])
        plant = G3 * G2 * G1
        pid = control.tf([Kd, Kp, Ki], [1, 0])
        closed_loop = control.feedback(pid * plant, 1)

        # 周波数軸と周波数応答
        omega = np.logspace(1, 5, 1000)
        mag_open, phase_open, _ = control.bode(plant, omega=omega, plot=False)
        mag_closed, phase_closed, _ = control.bode(closed_loop, omega=omega, plot=False)

        mag_open_db = 20 * np.log10(mag_open)
        mag_closed_db = 20 * np.log10(mag_closed)
        phase_open_deg = np.degrees(phase_open)
        phase_closed_deg = np.degrees(phase_closed)

        self.ax_mag.clear()
        self.ax_phase.clear()
        self.ax_mag.semilogx(omega, mag_open_db, label="開ループ")
        self.ax_mag.semilogx(omega, mag_closed_db, label="閉ループ")
        self.ax_mag.set_ylabel("ゲイン [dB]")
        self.ax_mag.set_title("ゲイン特性")
        self.ax_mag.grid(True, which='both')
        self.ax_mag.legend()

        self.ax_phase.semilogx(omega, phase_open_deg, label="開ループ")
        self.ax_phase.semilogx(omega, phase_closed_deg, label="閉ループ")
        self.ax_phase.set_ylabel("位相 [deg]")
        self.ax_phase.set_xlabel("周波数 [rad/s]")
        self.ax_phase.set_title("位相特性")
        self.ax_phase.grid(True, which='both')
        self.ax_phase.legend()

        # ゲイン・位相余裕表示
        gm_open, pm_open, _, _ = control.margin(plant)
        gm_db_open = 20 * np.log10(gm_open) if gm_open != np.inf else np.inf
        self.gm_open_label.config(text=f"開ループ GM: {gm_db_open:.2f} dB" if gm_open != np.inf else "開ループ GM: ∞ dB")
        self.pm_open_label.config(text=f"開ループ PM: {pm_open:.2f} °" if pm_open != np.inf else "開ループ PM: ∞ °")

        gm_closed, pm_closed, _, _ = control.margin(closed_loop)
        gm_db_closed = 20 * np.log10(gm_closed) if gm_closed != np.inf else np.inf
        self.gm_closed_label.config(text=f"閉ループ GM: {gm_db_closed:.2f} dB" if gm_closed != np.inf else "閉ループ GM: ∞ dB")
        self.pm_closed_label.config(text=f"閉ループ PM: {pm_closed:.2f} °" if pm_closed != np.inf else "閉ループ PM: ∞ °")

        self.canvas.draw()

    def on_close(self):
        plt.close(self.fig)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VCMControlApp(root)
    root.mainloop()