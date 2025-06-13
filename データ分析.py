import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class DataPreprocessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("機械学習・データ前処理統合アプリ")
        self.df = None
        self.history = []

        self.target_var = tk.StringVar()
        self.model_type = tk.StringVar(value="線形回帰")
        self.cv_folds = tk.IntVar(value=5)
        self.param_grid_str = tk.StringVar()

        self.build_gui()

    def build_gui(self):
        frame_file = tk.Frame(self.root)
        frame_file.pack(fill=tk.X)
        tk.Button(frame_file, text="CSV読み込み", command=self.load_csv).pack(side=tk.LEFT)
        tk.Button(frame_file, text="CSV保存", command=self.save_csv).pack(side=tk.LEFT)
        tk.Button(frame_file, text="元に戻す", command=self.undo).pack(side=tk.LEFT)

        frame_proc = tk.Frame(self.root)
        frame_proc.pack(fill=tk.X)
        tk.Button(frame_proc, text="欠損行削除", command=lambda: self.drop_na(mode='any')).pack(side=tk.LEFT)
        tk.Button(frame_proc, text="すべて欠損行削除", command=lambda: self.drop_na(mode='all')).pack(side=tk.LEFT)
        tk.Button(frame_proc, text="平均補完", command=lambda: self.fill_na('mean')).pack(side=tk.LEFT)
        tk.Button(frame_proc, text="中央値補完", command=lambda: self.fill_na('median')).pack(side=tk.LEFT)
        tk.Button(frame_proc, text="スケーリング", command=self.scale_data).pack(side=tk.LEFT)
        tk.Button(frame_proc, text="相関＋クラスタ表示", command=self.show_correlation).pack(side=tk.LEFT)

        self.tree_frame = tk.Frame(self.root)
        self.tree_frame.pack(fill=tk.BOTH, expand=True)
        self.tree_scroll_y = ttk.Scrollbar(self.tree_frame, orient="vertical")
        self.tree_scroll_x = ttk.Scrollbar(self.tree_frame, orient="horizontal")
        self.data_tree = ttk.Treeview(self.tree_frame, yscrollcommand=self.tree_scroll_y.set, xscrollcommand=self.tree_scroll_x.set)
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree_scroll_y.config(command=self.data_tree.yview)
        self.tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_scroll_x.config(command=self.data_tree.xview)
        self.tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        frame_graph = tk.Frame(self.root)
        frame_graph.pack(fill=tk.X)
        tk.Label(frame_graph, text="グラフ種類").pack(side=tk.LEFT)
        self.plot_type = ttk.Combobox(frame_graph, values=["ヒストグラム", "散布図", "折れ線"], width=10)
        self.plot_type.pack(side=tk.LEFT)
        self.plot_type.set("ヒストグラム")
        self.plot_x = ttk.Combobox(frame_graph, width=15)
        self.plot_x.pack(side=tk.LEFT)
        self.plot_y = ttk.Combobox(frame_graph, width=15)
        self.plot_y.pack(side=tk.LEFT)
        tk.Button(frame_graph, text="グラフ描画", command=self.draw_plot).pack(side=tk.LEFT)

        frame_ml = tk.Frame(self.root)
        frame_ml.pack(fill=tk.X)
        tk.Label(frame_ml, text="目的変数").pack(side=tk.LEFT)
        self.target_combo = ttk.Combobox(frame_ml, textvariable=self.target_var, width=15)
        self.target_combo.pack(side=tk.LEFT)
        tk.Label(frame_ml, text="モデル").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(frame_ml, textvariable=self.model_type, values=["線形回帰", "ロジスティック回帰", "ランダムフォレスト（回帰）", "ランダムフォレスト（分類）"], width=25)
        self.model_combo.pack(side=tk.LEFT)
        tk.Button(frame_ml, text="学習実行", command=self.run_model).pack(side=tk.LEFT)

        frame_feat = tk.Frame(self.root)
        frame_feat.pack(fill=tk.X)
        tk.Label(frame_feat, text="説明変数").pack(side=tk.LEFT)
        self.feature_listbox = tk.Listbox(frame_feat, selectmode=tk.MULTIPLE, exportselection=False, width=50, height=5)
        self.feature_listbox.pack(side=tk.LEFT)
        tk.Label(frame_feat, text="交差検証Fold").pack(side=tk.LEFT)
        tk.Entry(frame_feat, textvariable=self.cv_folds, width=5).pack(side=tk.LEFT)

        frame_grid = tk.Frame(self.root)
        frame_grid.pack(fill=tk.X)
        tk.Label(frame_grid, text="グリッドサーチ パラメータ辞書").pack(side=tk.LEFT)
        tk.Entry(frame_grid, textvariable=self.param_grid_str, width=50).pack(side=tk.LEFT)

        self.log_text = tk.Text(self.root, height=10)
        self.log_text.pack(fill=tk.BOTH)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.df = pd.read_csv(path)
            self.history = []
            self.update_treeview()
            self.update_column_options()
            self.log("CSVを読み込みました。")

    def save_csv(self):
        if self.df is not None:
            path = filedialog.asksaveasfilename(defaultextension=".csv")
            if path:
                self.df.to_csv(path, index=False)
                self.log("CSVを保存しました。")

    def undo(self):
        if self.history:
            self.df = self.history.pop()
            self.update_treeview()
            self.update_column_options()
            self.log("1つ前に戻しました。")

    def drop_na(self, mode='any'):
        if self.df is not None:
            self.history.append(self.df.copy())
            self.df.dropna(axis=0, how=mode, inplace=True)
            self.update_treeview()
            self.log(f"欠損行を削除（mode='{mode}'）")

    def fill_na(self, method='mean'):
        if self.df is not None:
            self.history.append(self.df.copy())
            if method == 'mean':
                self.df.fillna(self.df.mean(numeric_only=True), inplace=True)
            elif method == 'median':
                self.df.fillna(self.df.median(numeric_only=True), inplace=True)
            self.update_treeview()
            self.log(f"{method}で補完しました。")

    def scale_data(self):
        if self.df is not None:
            self.history.append(self.df.copy())
            scaler = StandardScaler()
            numeric_cols = self.df.select_dtypes(include=np.number).columns
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
            self.update_treeview()
            self.log("標準化しました。")

    def update_treeview(self):
        self.data_tree.delete(*self.data_tree.get_children())
        if self.df is None:
            return
        self.data_tree["columns"] = list(self.df.columns)
        self.data_tree["show"] = "headings"
        for col in self.df.columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        for _, row in self.df.iterrows():
            self.data_tree.insert("", "end", values=list(row))

    def update_column_options(self):
        if self.df is not None:
            cols = list(self.df.columns)
            self.target_combo['values'] = cols
            self.plot_x['values'] = cols
            self.plot_y['values'] = cols
            self.feature_listbox.delete(0, tk.END)
            for col in cols:
                self.feature_listbox.insert(tk.END, col)

    def show_correlation(self):
        if self.df is not None:
            corr = self.df.corr(numeric_only=True)
            sns.set(font="Yu Gothic")
            sns.clustermap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("相関クラスターマップ")
            plt.show()

    def draw_plot(self):
        kind = self.plot_type.get()
        x = self.plot_x.get()
        y = self.plot_y.get()
        if self.df is None or x not in self.df.columns:
            self.log("有効なX軸を選択してください。")
            return
        if kind == "ヒストグラム":
            self.df[x].hist(bins=30)
        elif kind == "散布図":
            if y in self.df.columns:
                sns.scatterplot(x=self.df[x], y=self.df[y])
        elif kind == "折れ線":
            if y in self.df.columns:
                plt.plot(self.df[x], self.df[y])
        plt.title(f"{x} vs {y}")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run_model(self):
        target = self.target_var.get()
        if not target or target not in self.df.columns:
            self.log("目的変数が不正です")
            return
        X = self.df.drop(columns=[target])
        y = self.df[target]
        X = X.select_dtypes(include=[np.number])
        if X.empty or y.isnull().any():
            self.log("データが不正（欠損 or 数値でない）")
            return
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        model_name = self.model_type.get()
        model = {
            "線形回帰": LinearRegression(),
            "ロジスティック回帰": LogisticRegression(max_iter=1000),
            "ランダムフォレスト（回帰）": RandomForestRegressor(),
            "ランダムフォレスト（分類）": RandomForestClassifier()
        }.get(model_name)
        if model is None:
            self.log("未対応モデルです")
            return
        param_str = self.param_grid_str.get().strip()
        if param_str:
            try:
                grid = eval(param_str)
                model = GridSearchCV(model, grid, cv=self.cv_folds.get())
            except:
                self.log("パラメータ形式エラー")
                return
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if y.dtype.kind in "bifc":
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            self.log(f"RMSE: {rmse:.3f} / R²: {r2:.3f}")
        else:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            self.log(f"Accuracy: {acc:.3f} / F1: {f1:.3f}")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1300x800")
    app = DataPreprocessorApp(root)
    root.mainloop()
