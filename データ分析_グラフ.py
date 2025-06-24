import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix, classification_report

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
        self.scaler_type = tk.StringVar(value="StandardScaler")
        self.encoder_type = tk.StringVar(value="OneHot")

        self.build_gui()

    def build_gui(self):
        frame_file = tk.Frame(self.root)
        frame_file.pack(fill=tk.X)
        tk.Button(frame_file, text="CSV読み込み", command=self.load_csv).pack(side=tk.LEFT)
        tk.Button(frame_file, text="CSV保存", command=self.save_csv).pack(side=tk.LEFT)
        tk.Button(frame_file, text="元に戻す", command=self.undo).pack(side=tk.LEFT)
        tk.Button(frame_file, text="EDA実行", command=self.show_eda_window).pack(side=tk.LEFT)

        frame_select = tk.Frame(self.root)
        frame_select.pack(fill=tk.X)
        tk.Label(frame_select, text="処理対象列（複数選択可）").pack(side=tk.LEFT)
        self.col_listbox = tk.Listbox(frame_select, selectmode=tk.MULTIPLE, exportselection=False, width=80, height=5)
        self.col_listbox.pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(frame_select, text="全選択", command=self.select_all_columns).pack(side=tk.LEFT)
        tk.Button(frame_select, text="全解除", command=self.clear_all_columns).pack(side=tk.LEFT)

        frame_proc = tk.Frame(self.root)
        frame_proc.pack(fill=tk.X)
        tk.Button(frame_proc, text="欠損行削除", command=self.drop_na_selected).pack(side=tk.LEFT)
        tk.Button(frame_proc, text="平均補完", command=self.fill_na_mean).pack(side=tk.LEFT)
        tk.Label(frame_proc, text="スケーリング種類").pack(side=tk.LEFT)
        self.scaler_combo = ttk.Combobox(frame_proc, textvariable=self.scaler_type, values=["StandardScaler", "MinMaxScaler"], width=15)
        self.scaler_combo.pack(side=tk.LEFT)
        tk.Button(frame_proc, text="スケーリング", command=self.scale_selected).pack(side=tk.LEFT)
        tk.Label(frame_proc, text="エンコード").pack(side=tk.LEFT)
        self.encoder_combo = ttk.Combobox(frame_proc, textvariable=self.encoder_type, values=["OneHot", "Label"], width=10)
        self.encoder_combo.pack(side=tk.LEFT)
        tk.Button(frame_proc, text="エンコード実行", command=self.encode_selected).pack(side=tk.LEFT)
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

        frame_ml = tk.Frame(self.root)
        frame_ml.pack(fill=tk.X)
        tk.Label(frame_ml, text="目的変数").pack(side=tk.LEFT)
        self.target_combo = ttk.Combobox(frame_ml, textvariable=self.target_var, width=15)
        self.target_combo.pack(side=tk.LEFT)
        tk.Label(frame_ml, text="モデル").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(frame_ml, textvariable=self.model_type, values=[
            "線形回帰", "ロジスティック回帰", "ランダムフォレスト（回帰）", "ランダムフォレスト（分類）"
        ], width=25)
        self.model_combo.pack(side=tk.LEFT)
        tk.Button(frame_ml, text="学習実行", command=self.run_model).pack(side=tk.LEFT)

        frame_feat = tk.Frame(self.root)
        frame_feat.pack(fill=tk.X)
        tk.Label(frame_feat, text="説明変数").pack(side=tk.LEFT)
        self.feature_listbox = tk.Listbox(frame_feat, selectmode=tk.MULTIPLE, exportselection=False, width=50, height=5)
        self.feature_listbox.pack(side=tk.LEFT)
        tk.Button(frame_feat, text="全選択", command=self.select_all_features).pack(side=tk.LEFT)
        tk.Button(frame_feat, text="全解除", command=self.clear_all_features).pack(side=tk.LEFT)
        tk.Label(frame_feat, text="交差検証Fold").pack(side=tk.LEFT)
        tk.Entry(frame_feat, textvariable=self.cv_folds, width=5).pack(side=tk.LEFT)

        frame_grid = tk.Frame(self.root)
        frame_grid.pack(fill=tk.X)
        tk.Label(frame_grid, text="グリッドサーチ パラメータ辞書").pack(side=tk.LEFT)
        tk.Entry(frame_grid, textvariable=self.param_grid_str, width=50).pack(side=tk.LEFT)

        self.log_text = tk.Text(self.root, height=10)
        self.log_text.pack(fill=tk.BOTH)

    def log(self, msg):
        self.log_text.insert(tk.END, str(msg) + "\n")
        self.log_text.see(tk.END)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.df = pd.read_csv(path)
            self.history = []
            self.update_treeview()
            self.update_column_options()
            self.log("✅ CSVを読み込みました。")

    def save_csv(self):
        if self.df is not None:
            path = filedialog.asksaveasfilename(defaultextension=".csv")
            if path:
                self.df.to_csv(path, index=False)
                self.log("💾 CSVを保存しました。")

    def undo(self):
        if self.history:
            self.df = self.history.pop()
            self.update_treeview()
            self.update_column_options()
            self.log("↩️ 1つ前に戻しました。")

    def get_selected_columns(self):
        return [self.col_listbox.get(i) for i in self.col_listbox.curselection()]

    def drop_na_selected(self):
        cols = self.get_selected_columns()
        if not cols:
            self.log("⚠️ 列を選んでください。")
            return
        self.history.append(self.df.copy())
        self.df.dropna(subset=cols, inplace=True)
        self.update_treeview()
        self.log(f"🧹 欠損行を削除（{cols}）")

    def fill_na_mean(self):
        cols = self.get_selected_columns()
        if not cols:
            self.log("⚠️ 列を選んでください。")
            return
        self.history.append(self.df.copy())
        for col in cols:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        self.update_treeview()
        self.log(f"🔧 平均値で補完（{cols}）")

    def scale_selected(self):
        cols = self.get_selected_columns()
        if not cols:
            self.log("⚠️ 列を選んでください。")
            return
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(self.df[col])]
        if not numeric_cols:
            self.log("⚠️ 数値列がありません。")
            return
        self.history.append(self.df.copy())
        scaler = StandardScaler() if self.scaler_type.get() == "StandardScaler" else MinMaxScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        self.update_treeview()
        self.log(f"📐 スケーリング完了: {numeric_cols}")

    def encode_selected(self):
        cols = self.get_selected_columns()
        if not cols:
            self.log("⚠️ 列を選んでください。")
            return
        self.history.append(self.df.copy())
        for col in cols:
            if self.encoder_type.get() == "OneHot":
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
            else:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
        self.update_treeview()
        self.log(f"🔤 エンコード完了: {cols}")

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
            self.target_combo["values"] = cols
            self.col_listbox.delete(0, tk.END)
            self.feature_listbox.delete(0, tk.END)
            for col in cols:
                self.col_listbox.insert(tk.END, col)
                self.feature_listbox.insert(tk.END, col)

    def run_model(self):
        target = self.target_var.get()
        if not target or target not in self.df.columns:
            self.log("⚠️ 目的変数を選択してください。")
            return
        features = [self.feature_listbox.get(i) for i in self.feature_listbox.curselection()]
        if not features:
            self.log("⚠️ 説明変数を選んでください。")
            return
        X = self.df[features].select_dtypes(include=[np.number])
        y = self.df[target]
        if X.empty:
            self.log("⚠️ 数値の説明変数が必要です。")
            return
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_dict = {
            "線形回帰": LinearRegression(),
            "ロジスティック回帰": LogisticRegression(max_iter=1000),
            "ランダムフォレスト（回帰）": RandomForestRegressor(),
            "ランダムフォレスト（分類）": RandomForestClassifier()
        }
        model = model_dict.get(self.model_type.get())
        if model is None:
            self.log("⚠️ 未対応のモデルです。")
            return
        param_str = self.param_grid_str.get()
        if param_str:
            try:
                param_dict = eval(param_str)
                model = GridSearchCV(model, param_dict, cv=self.cv_folds.get())
            except:
                self.log("⚠️ パラメータ形式が不正です。")
                return
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if y.dtype.kind in "bifc":
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            self.log(f"📈 回帰モデル - RMSE: {rmse:.3f} / R2: {r2:.3f}")
            self.plot_regression_result(y_test, y_pred)
        else:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            self.log(f"📊 分類モデル - Accuracy: {acc:.3f} / F1: {f1:.3f}")
            self.log(classification_report(y_test, y_pred))
            self.plot_confusion_matrix(cm, labels=np.unique(y_test))

    def plot_confusion_matrix(self, cm, labels):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title("混同行列")
        ax.set_xlabel("予測値")
        ax.set_ylabel("実測値")
        plt.tight_layout()
        plt.show()

    def plot_regression_result(self, y_true, y_pred):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_true, y_pred, alpha=0.6)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
        ax.set_xlabel("実測値")
        ax.set_ylabel("予測値")
        ax.set_title("回帰結果の可視化")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def show_eda_window(self):
        if self.df is None:
            self.log("❌ データが読み込まれていません。")
            return
        eda_win = tk.Toplevel(self.root)
        eda_win.title("📊 EDA分析結果")
        eda_win.geometry("900x600")
        text = tk.Text(eda_win, wrap=tk.NONE)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, "📊 === 基本統計量 ===\n")
        text.insert(tk.END, str(self.df.describe(include='all')) + "\n\n")
        text.insert(tk.END, "🧩 === 欠損値の数 ===\n")
        text.insert(tk.END, str(self.df.isnull().sum()) + "\n\n")
        text.insert(tk.END, "🔢 === ユニーク値の数 ===\n")
        text.insert(tk.END, str(self.df.nunique()) + "\n\n")
        text.insert(tk.END, "🔠 === カテゴリ列の頻出値 ===\n")
        cat_cols = self.df.select_dtypes(include='object').columns
        for col in cat_cols:
            text.insert(tk.END, f"\n[{col}]\n")
            text.insert(tk.END, str(self.df[col].value_counts().head(5)) + "\n")
        text.config(state=tk.DISABLED)

    def show_correlation(self):
        if self.df is None:
            self.log("❌ データが読み込まれていません。")
            return
        corr = self.df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("相関マトリクス")
        plt.tight_layout()
        plt.show()

    def select_all_columns(self):
        self.col_listbox.select_set(0, tk.END)

    def clear_all_columns(self):
        self.col_listbox.select_clear(0, tk.END)

    def select_all_features(self):
        self.feature_listbox.select_set(0, tk.END)

    def clear_all_features(self):
        self.feature_listbox.select_clear(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1400x950")
    app = DataPreprocessorApp(root)
    root.mainloop()
