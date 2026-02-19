import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import joblib
import numpy as np
import torch
import torch.nn as nn


# 工程路径与输出目录
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'outputs'


class BPNet(nn.Module):
    # 与训练脚本一致：单隐层 BP 网络
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)


class WaterQualityUI:
    def __init__(self, root: tk.Tk):
        # 基础窗口设置
        self.root = root
        self.root.title('水质神经网络工具箱')
        self.root.geometry('900x640')

        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        self.cfg = None
        self.input_entries = []

        self._build_layout()
        self._render_input_fields([])

    def _build_layout(self):
        # 顶部功能按钮：训练 / 导出 / 加载
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill=tk.X)

        ttk.Button(top, text='1) 训练模型(new.py)', command=self.run_training).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text='2) 导出参数(export_to_c.py)', command=self.run_export).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text='3) 加载模型', command=self.load_model).pack(side=tk.LEFT, padx=6)

        status_frame = ttk.Frame(self.root, padding=(12, 0))
        status_frame.pack(fill=tk.X)
        self.status_var = tk.StringVar(value='状态: 未加载模型')
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W)

        input_box = ttk.LabelFrame(self.root, text='预测输入', padding=12)
        input_box.pack(fill=tk.X, padx=12, pady=8)

        self.input_container = ttk.Frame(input_box)
        self.input_container.pack(fill=tk.X)

        predict_btn = ttk.Button(input_box, text='执行预测', command=self.predict)
        predict_btn.pack(anchor=tk.W, pady=(10, 0))

        self.predict_result_var = tk.StringVar(value='预测结果: --')
        ttk.Label(input_box, textvariable=self.predict_result_var, font=('微软雅黑', 11, 'bold')).pack(anchor=tk.W, pady=6)

        log_box = ttk.LabelFrame(self.root, text='运行日志', padding=8)
        log_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        self.log_text = tk.Text(log_box, wrap=tk.WORD, height=18)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        footer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        footer.pack(fill=tk.X)
        ttk.Label(footer, text=f'输出目录: {OUTPUT_DIR}').pack(anchor=tk.W)

    def _append_log(self, text: str):
        # 在日志框中追加文本
        self.log_text.insert(tk.END, text + '\n')
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _run_script(self, script_name: str):
        # 统一执行外部脚本并捕获输出
        script_path = BASE_DIR / script_name
        if not script_path.exists():
            self._append_log(f'[错误] 未找到脚本: {script_path}')
            return False

        python_exe = sys.executable
        self._append_log(f'>>> 运行: {python_exe} {script_name}')
        try:
            proc = subprocess.run(
                [python_exe, str(script_path)],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
            )
            if proc.stdout:
                self._append_log(proc.stdout.strip())
            if proc.stderr:
                self._append_log('[stderr]')
                self._append_log(proc.stderr.strip())
            if proc.returncode != 0:
                self._append_log(f'[失败] 退出码: {proc.returncode}')
                return False
            self._append_log('[完成]')
            return True
        except Exception as exc:
            self._append_log(f'[异常] {exc}')
            return False

    def run_training(self):
        # 点击按钮后执行训练脚本
        ok = self._run_script('new.py')
        if ok:
            self.status_var.set('状态: 训练完成，请点击“加载模型”')

    def run_export(self):
        # 点击按钮后导出 C 参数头文件
        ok = self._run_script('export_to_c.py')
        if ok:
            self.status_var.set('状态: 参数导出完成')

    def _render_input_fields(self, fields):
        # 根据模型输入字段动态生成输入框
        for child in self.input_container.winfo_children():
            child.destroy()

        self.input_entries = []
        if not fields:
            ttk.Label(self.input_container, text='请先点击“加载模型”').pack(anchor=tk.W)
            return

        for i, name in enumerate(fields):
            row = ttk.Frame(self.input_container)
            row.pack(fill=tk.X, pady=4)
            ttk.Label(row, text=f'{name}:', width=20).pack(side=tk.LEFT)
            entry = ttk.Entry(row)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.input_entries.append((name, entry))

    def load_model(self):
        # 从 outputs 目录加载模型及归一化器
        config_path = OUTPUT_DIR / 'model_config.json'
        model_path = OUTPUT_DIR / 'water_quality_model.pth'
        scaler_x_path = OUTPUT_DIR / 'scaler_x.pkl'
        scaler_y_path = OUTPUT_DIR / 'scaler_y.pkl'

        required = [config_path, model_path, scaler_x_path, scaler_y_path]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            messagebox.showerror('加载失败', '缺少以下文件:\n' + '\n'.join(missing))
            self.status_var.set('状态: 加载失败，文件不完整')
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.cfg = json.load(f)

            self.model = BPNet(self.cfg['input_size'], self.cfg['hidden_size'], self.cfg['output_size'])
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

            self.scaler_x = joblib.load(scaler_x_path)
            self.scaler_y = joblib.load(scaler_y_path)

            # 根据输入字段动态刷新输入控件
            input_cols = self.cfg.get('input_cols', [])
            self._render_input_fields(input_cols)
            target_cols = self.cfg.get('target_cols')
            if not target_cols:
                target_cols = [self.cfg.get('target_col', 'unknown')]
            self.cfg['target_cols'] = target_cols
            self.status_var.set(f"状态: 模型已加载，输出字段={', '.join(target_cols)}")
            self._append_log('[模型加载成功]')
        except Exception as exc:
            messagebox.showerror('加载失败', str(exc))
            self.status_var.set('状态: 模型加载异常')

    def predict(self):
        # 执行一次前向推理并显示双输出结果
        if self.model is None or self.scaler_x is None or self.scaler_y is None or self.cfg is None:
            messagebox.showwarning('未加载模型', '请先点击“加载模型”。')
            return

        try:
            values = []
            for name, entry in self.input_entries:
                text = entry.get().strip()
                if text == '':
                    raise ValueError(f'输入项 {name} 不能为空')
                values.append(float(text))

            arr = np.array([values], dtype=np.float32)
            x_scaled = self.scaler_x.transform(arr)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

            with torch.no_grad():
                y_scaled = self.model(x_tensor).cpu().numpy()

            y_real = self.scaler_y.inverse_transform(y_scaled)
            pred = y_real[0]
            target_cols = self.cfg.get('target_cols', [self.cfg.get('target_col', 'target')])
            result_text = '，'.join([f'{name} = {pred[i]:.6f}' for i, name in enumerate(target_cols)])

            self.predict_result_var.set(f'预测结果: {result_text}')
            self._append_log(f'[预测] {result_text}')
        except Exception as exc:
            messagebox.showerror('预测失败', str(exc))


def main():
    # 程序入口
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    app = WaterQualityUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
