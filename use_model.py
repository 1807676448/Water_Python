import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


# 工程根目录与统一输出目录
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'outputs'

# ======================== 运行参数宏定义（集中修改） ========================
# 模型系统文件名
MODEL_FILE = 'water_quality_model.pth'
SCALER_X_FILE = 'scaler_x.pkl'
SCALER_Y_FILE = 'scaler_y.pkl'
MODEL_CONFIG_FILE = 'model_config.json'

# 交互运行参数
QUIT_COMMAND = 'q'  # 输入该字符串可退出预测循环
INPUT_SPLIT_MODE = None  # None 表示按任意空白字符分割（空格/Tab）
PREDICT_PRINT_PRECISION = 6  # 预测值打印小数位

# 输出控制
PRINT_TRAINING_PARAM_EXPLANATION = True  # 启动时打印训练参数说明

# ======================== 训练参数宏定义（说明/默认值） ========================
# 说明：use_model.py 是推理脚本，训练参数实际来源为 model_config.json。
# 下面宏定义用于“集中说明 + 缺省兜底”，当配置文件缺失对应字段时会回退到这些默认值。
TRAIN_INPUT_SIZE = 3
TRAIN_HIDDEN_SIZE = 8
TRAIN_OUTPUT_SIZE = 2
TRAIN_ACTIVATION_HIDDEN = 'tansig'
TRAIN_ACTIVATION_OUTPUT = 'purelin'
TRAIN_NORMALIZATION = 'MinMaxScaler(-1,1)'
TRAIN_ALGO = 'gd'
TRAIN_USE_PSO_INIT = True
TRAIN_SEED = 42


class BPNet(nn.Module):
    # BP 网络：单隐层 tansig + 线性输出
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)


def load_model_system():
    # 检查模型与归一化参数是否存在
    required = [
        OUTPUT_DIR / MODEL_FILE,
        OUTPUT_DIR / SCALER_X_FILE,
        OUTPUT_DIR / SCALER_Y_FILE,
        OUTPUT_DIR / MODEL_CONFIG_FILE,
    ]
    for path in required:
        if not path.exists():
            print(f'错误: 未找到 {path}，请先运行 new.py 训练模型。')
            return None, None, None, None

    with open(OUTPUT_DIR / MODEL_CONFIG_FILE, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # 训练参数兜底：配置缺失时使用宏定义默认值
    cfg.setdefault('input_size', TRAIN_INPUT_SIZE)
    cfg.setdefault('hidden_size', TRAIN_HIDDEN_SIZE)
    cfg.setdefault('output_size', TRAIN_OUTPUT_SIZE)
    cfg.setdefault('activation_hidden', TRAIN_ACTIVATION_HIDDEN)
    cfg.setdefault('activation_output', TRAIN_ACTIVATION_OUTPUT)
    cfg.setdefault('normalization', TRAIN_NORMALIZATION)
    cfg.setdefault('train_algo', TRAIN_ALGO)
    cfg.setdefault('use_pso_init', TRAIN_USE_PSO_INIT)
    cfg.setdefault('seed', TRAIN_SEED)

    # 兼容旧配置：若无 target_cols，则回退为单输出配置
    target_cols = cfg.get('target_cols')
    if not target_cols:
        target_cols = [cfg.get('target_col', 'target')]

    model = BPNet(cfg['input_size'], cfg['hidden_size'], cfg['output_size'])
    model.load_state_dict(torch.load(OUTPUT_DIR / MODEL_FILE))
    model.eval()

    scaler_x = joblib.load(OUTPUT_DIR / SCALER_X_FILE)
    scaler_y = joblib.load(OUTPUT_DIR / SCALER_Y_FILE)

    cfg['target_cols'] = target_cols
    return model, scaler_x, scaler_y, cfg


def predict_multi(model, scaler_x, scaler_y, values):
    # 输入归一化 -> 模型推理 -> 输出反归一化
    arr = np.array([values], dtype=np.float32)
    x_scaled = scaler_x.transform(arr)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_scaled = model(x_tensor).cpu().numpy()

    y_real = scaler_y.inverse_transform(y_scaled)
    return y_real[0]


def evaluate_regression(y_true, y_pred):
    # 评价函数作用：
    # - 用统一指标量化预测质量，便于比较不同训练参数下模型效果。
    # - 常用于训练后/验证集测试阶段，不用于在线推理本身。
    # 返回指标含义：
    # - mse: 均方误差，越小越好。
    # - rmse: 均方根误差，量纲与目标值一致，越小越好。
    # - mae: 平均绝对误差，越小越好。
    # - r2: 拟合优度，越接近 1 越好。
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true 与 y_pred 形状不一致，无法评价。')

    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }


def print_training_param_explanation(cfg):
    # 训练参数作用与结果说明：
    # - input_size/output_size：决定输入输出维度，影响网络结构是否匹配数据。
    # - hidden_size：隐藏层神经元数量；偏小可能欠拟合，偏大可能过拟合。
    # - activation_hidden/activation_output：激活函数类型，影响非线性表达能力与输出范围。
    # - normalization：训练与推理统一归一化方式，保证输入输出尺度一致。
    # - train_algo：训练优化方式（如 gd），影响收敛速度和稳定性。
    # - use_pso_init：是否使用 PSO 初始化权重，常用于改善初始点和收敛表现。
    print('\n--- 训练参数说明 ---')
    print(f"input_size={cfg.get('input_size')}：输入特征维度")
    print(f"hidden_size={cfg.get('hidden_size')}：隐藏层规模，平衡拟合能力与泛化")
    print(f"output_size={cfg.get('output_size')}：输出目标维度")
    print(
        f"activation_hidden={cfg.get('activation_hidden')}，"
        f"activation_output={cfg.get('activation_output')}：网络非线性与输出形式"
    )
    print(f"normalization={cfg.get('normalization')}：训练/推理数据尺度一致性")
    print(f"train_algo={cfg.get('train_algo')}：优化过程与收敛特性")
    print(f"use_pso_init={cfg.get('use_pso_init')}：是否用 PSO 改善初始权重")

    print('\n--- 结果与评价建议 ---')
    print('结果通常用 MSE/RMSE/MAE/R2 综合判断：误差类指标越小越好，R2 越接近 1 越好。')
    print('建议在验证集或测试集上使用 evaluate_regression() 对比不同训练参数组合。')


def main():
    # 加载模型系统
    model, scaler_x, scaler_y, cfg = load_model_system()
    if model is None:
        return

    input_cols = cfg['input_cols']
    target_cols = cfg['target_cols']

    print('--- 水质 BP 模型预测 ---')
    print('输入字段:', ', '.join(input_cols))
    print('输出字段:', ', '.join(target_cols))
    if PRINT_TRAINING_PARAM_EXPLANATION:
        print_training_param_explanation(cfg)

    while True:
        try:
            raw = input("\n请输入输入值(空格分隔，输入 q 退出): ")
            if raw.strip().lower() == QUIT_COMMAND:
                break

            parts = raw.strip().split(INPUT_SPLIT_MODE)
            if len(parts) != len(input_cols):
                print(f'输入数量错误，需要 {len(input_cols)} 个值。')
                continue

            # 执行双输出预测并格式化打印
            values = [float(v) for v in parts]
            pred = predict_multi(model, scaler_x, scaler_y, values)
            text = '，'.join(
                [f'{name}: {pred[i]:.{PREDICT_PRINT_PRECISION}f}' for i, name in enumerate(target_cols)]
            )
            print(f'预测结果 -> {text}')
        except ValueError:
            print('输入格式错误，请输入数字。')
        except Exception as exc:
            print(f'发生错误: {exc}')


if __name__ == '__main__':
    main()
