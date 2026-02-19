import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# 工程根目录与统一输出目录
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================== 运行与训练参数宏定义（集中修改） ========================
# 文件与日志
LOG_FILE = 'training_log.txt'
MODEL_FILE = 'water_quality_model.pth'
SCALER_X_FILE = 'scaler_x.pkl'
SCALER_Y_FILE = 'scaler_y.pkl'
MODEL_CONFIG_FILE = 'model_config.json'

# 随机种子（影响数据划分、参数初始化与训练可复现性）
SEED = 42

# 数据与训练主参数
DEFAULT_EXCEL_FILE = 'shuizhi.xlsx'
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SCENARIO = 'custom'  # 'temperature_compensation' / 'turbidity_prediction' / 'custom'
DEFAULT_INPUT_COLS = ('254', '550', 'tem')
DEFAULT_TARGET_COLS = ('cod', 'uv254')
DEFAULT_HIDDEN_SIZE = 8
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_MAX_EPOCHS = 10000
DEFAULT_TARGET_MSE = 1e-5
DEFAULT_TRAIN_ALGO = 'gd'  # 'gd' or 'trainlm_like'

# PSO 初始化参数（仅当 use_pso_init=True 生效）
DEFAULT_USE_PSO_INIT = True
DEFAULT_PSO_PARTICLES = 20
DEFAULT_PSO_ITERS = 60
DEFAULT_PSO_W_MAX = 0.9
DEFAULT_PSO_W_MIN = 0.4
DEFAULT_PSO_C1 = 1.49445
DEFAULT_PSO_C2 = 1.49445

# 模型结构/预处理元信息（用于保存配置）
ACTIVATION_HIDDEN = 'tansig'
ACTIVATION_OUTPUT = 'purelin'
NORMALIZATION_METHOD = 'MinMaxScaler(-1,1)'


class Logger:
    # 训练日志同时输出到终端与文件
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(OUTPUT_DIR / LOG_FILE)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class TrainConfig:
    excel_file: str = DEFAULT_EXCEL_FILE
    test_size: float = DEFAULT_TEST_SIZE

    # 场景: 'temperature_compensation' / 'turbidity_prediction' / 'custom'
    scenario: str = DEFAULT_SCENARIO

    # custom 场景默认双输出：COD + UV254
    input_cols: tuple = DEFAULT_INPUT_COLS
    target_cols: tuple = DEFAULT_TARGET_COLS

    hidden_size: int = DEFAULT_HIDDEN_SIZE

    learning_rate: float = DEFAULT_LEARNING_RATE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    target_mse: float = DEFAULT_TARGET_MSE

    train_algo: str = DEFAULT_TRAIN_ALGO  # 'gd' or 'trainlm_like'

    use_pso_init: bool = DEFAULT_USE_PSO_INIT
    pso_particles: int = DEFAULT_PSO_PARTICLES
    pso_iters: int = DEFAULT_PSO_ITERS
    pso_w_max: float = DEFAULT_PSO_W_MAX
    pso_w_min: float = DEFAULT_PSO_W_MIN
    pso_c1: float = DEFAULT_PSO_C1
    pso_c2: float = DEFAULT_PSO_C2


class BPNet(nn.Module):
    # BP 网络：输入 -> tansig 隐层 -> 线性输出层
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)


def resolve_columns(df: pd.DataFrame, cfg: TrainConfig):
    # 根据场景解析输入/输出字段与网络规模
    cols = df.columns.astype(str).str.strip().tolist()

    if cfg.scenario == 'temperature_compensation':
        input_cols = ['tem', 'light_power', 'dark_current']
        target_cols = ['cod']
        hidden_size = 4
    elif cfg.scenario == 'turbidity_prediction':
        input_cols = ['scatter_voltage', 'transmit_voltage', 'ratio']
        target_cols = ['turbidity']
        hidden_size = 14
    else:
        input_cols = list(cfg.input_cols)
        target_cols = list(cfg.target_cols)
        hidden_size = cfg.hidden_size

    missing = [c for c in (input_cols + target_cols) if c not in cols]
    if missing:
        raise ValueError(
            f"数据列缺失: {missing}。当前可用列: {cols}。"
            "\n请修改 TrainConfig 中的 scenario/input_cols/target_cols。"
        )

    return input_cols, target_cols, hidden_size


def flatten_params(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])


def assign_params(model: nn.Module, flat_params: np.ndarray):
    cursor = 0
    for p in model.parameters():
        numel = p.numel()
        block = flat_params[cursor:cursor + numel].reshape(p.shape)
        p.data = torch.from_numpy(block).float()
        cursor += numel


def pso_initialize(model: BPNet, x_train: np.ndarray, y_train: np.ndarray, cfg: TrainConfig):
    # 使用 PSO 搜索更优初始权重，提升 BP 收敛与泛化
    dim = flatten_params(model).shape[0]
    n = cfg.pso_particles

    pos = np.random.uniform(-1.0, 1.0, size=(n, dim)).astype(np.float32)
    vel = np.zeros((n, dim), dtype=np.float32)

    pbest_pos = pos.copy()
    pbest_score = np.full(n, np.inf, dtype=np.float64)

    gbest_pos = None
    gbest_score = np.inf

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    criterion = nn.MSELoss()

    def fitness(candidate):
        assign_params(model, candidate)
        with torch.no_grad():
            pred = model(x_tensor)
            return criterion(pred, y_tensor).item()

    for i in range(n):
        score = fitness(pos[i])
        pbest_score[i] = score
        if score < gbest_score:
            gbest_score = score
            gbest_pos = pos[i].copy()

    print(f"[PSO] 初始全局最优 MSE: {gbest_score:.8f}")

    for t in range(cfg.pso_iters):
        w = cfg.pso_w_max - (cfg.pso_w_max - cfg.pso_w_min) * ((t / max(cfg.pso_iters - 1, 1)) ** 2)
        r1 = np.random.rand(n, dim).astype(np.float32)
        r2 = np.random.rand(n, dim).astype(np.float32)

        vel = w * vel + cfg.pso_c1 * r1 * (pbest_pos - pos) + cfg.pso_c2 * r2 * (gbest_pos - pos)
        pos = np.clip(pos + vel, -2.0, 2.0)

        for i in range(n):
            score = fitness(pos[i])
            if score < pbest_score[i]:
                pbest_score[i] = score
                pbest_pos[i] = pos[i].copy()
            if score < gbest_score:
                gbest_score = score
                gbest_pos = pos[i].copy()

        if (t + 1) % 10 == 0 or t == cfg.pso_iters - 1:
            print(f"[PSO] Iter {t + 1:03d}/{cfg.pso_iters}, gbest MSE: {gbest_score:.8f}")

    assign_params(model, gbest_pos)
    print(f"[PSO] 完成，使用最优参数初始化 BP 网络，MSE: {gbest_score:.8f}")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # 评价函数之一：MAPE（平均绝对百分比误差）
    # 作用：衡量相对误差大小，便于不同量纲/量级数据比较。
    # 结果解释：越小越好，0% 表示完全拟合。
    eps = 1e-8
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # 评价函数作用：统一输出回归指标，便于比较不同训练参数组合（学习率、隐层大小、是否PSO等）效果。
    # 返回指标解释：
    # - mse：均方误差，越小越好；对大误差更敏感。
    # - rmse：均方根误差，量纲与原始目标一致，越小越好。
    # - r2：拟合优度，越接近 1 越好。
    # - mape：平均绝对百分比误差，越小越好。
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    mape_value = mape(y_true, y_pred)
    return {
        'mse': float(mse),
        'rmse': rmse,
        'r2': float(r2),
        'mape': mape_value,
    }


def main():
    cfg = TrainConfig()
    print('--- 训练参数说明 ---')
    print(f'scenario={cfg.scenario}：训练场景，决定默认输入输出列与部分网络设置')
    print(f'hidden_size={cfg.hidden_size}：隐藏层规模，偏小易欠拟合，偏大易过拟合')
    print(f'learning_rate={cfg.learning_rate}：学习率，影响收敛速度与稳定性')
    print(f'max_epochs={cfg.max_epochs}：最大训练轮数，决定训练上限')
    print(f'target_mse={cfg.target_mse}：早停目标误差，达到后提前停止')
    print(f'train_algo={cfg.train_algo}：优化算法（gd 或 trainlm_like）')
    print(f'use_pso_init={cfg.use_pso_init}：是否先用 PSO 搜索更优初始权重')
    print('结果建议：综合关注 MSE/RMSE/R2/MAPE；误差类越小越好，R2 越接近 1 越好。\n')

    # 1) 读取数据
    print('日志系统已启动，输出将保存到 outputs/training_log.txt')
    print('读取数据中...')

    excel_path = Path(cfg.excel_file)
    if not excel_path.is_absolute():
        excel_path = BASE_DIR / excel_path

    df = pd.read_excel(excel_path)
    df.columns = df.columns.astype(str).str.strip()
    print('检测到列名:', df.columns.tolist())

    input_cols, target_cols, hidden_size = resolve_columns(df, cfg)
    output_size = len(target_cols)

    print(f'输入列: {input_cols}')
    print(f'输出列: {target_cols}')
    print(f'隐含层神经元: {hidden_size}')

    # 2) 划分训练/测试并做 MinMax 归一化
    x_raw = df[input_cols].values.astype(np.float32)
    y_raw = df[target_cols].values.astype(np.float32)

    x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(
        x_raw, y_raw, test_size=cfg.test_size, random_state=SEED
    )

    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    x_train = scaler_x.fit_transform(x_train_raw)
    x_test = scaler_x.transform(x_test_raw)
    y_train = scaler_y.fit_transform(y_train_raw)
    y_test = scaler_y.transform(y_test_raw)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    # 3) 构建模型
    model = BPNet(input_size=len(input_cols), hidden_size=hidden_size, output_size=output_size)

    # 4) 可选 PSO 初始化
    if cfg.use_pso_init:
        print('开始 PSO 初始化...')
        pso_initialize(model, x_train, y_train, cfg)

    # 5) BP 训练
    criterion = nn.MSELoss()
    print('开始 BP 训练...')

    if cfg.train_algo == 'trainlm_like':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=20)

        def closure():
            optimizer.zero_grad()
            pred = model(x_train_tensor)
            loss = criterion(pred, y_train_tensor)
            loss.backward()
            return loss

        for epoch in range(cfg.max_epochs):
            loss = optimizer.step(closure)
            current = float(loss.item())
            if (epoch + 1) % 50 == 0:
                print(f'Epoch {epoch + 1:04d}/{cfg.max_epochs}, Train MSE: {current:.8f}')
            if current <= cfg.target_mse:
                print(f'达到目标误差，提前停止。epoch={epoch + 1}, mse={current:.8f}')
                break
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
        for epoch in range(cfg.max_epochs):
            model.train()
            pred = model(x_train_tensor)
            loss = criterion(pred, y_train_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current = float(loss.item())
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1:04d}/{cfg.max_epochs}, Train MSE: {current:.8f}')
            if current <= cfg.target_mse:
                print(f'达到目标误差，提前停止。epoch={epoch + 1}, mse={current:.8f}')
                break

    # 6) 评估（总体 + 分输出）
    model.eval()
    with torch.no_grad():
        train_pred_s = model(x_train_tensor).cpu().numpy()
        test_pred_s = model(x_test_tensor).cpu().numpy()

    train_pred = scaler_y.inverse_transform(train_pred_s)
    test_pred = scaler_y.inverse_transform(test_pred_s)
    y_train_real = scaler_y.inverse_transform(y_train)
    y_test_real = scaler_y.inverse_transform(y_test)

    train_metrics = evaluate_regression(y_train_real, train_pred)
    test_metrics = evaluate_regression(y_test_real, test_pred)

    print('\n========== 模型总体评估 ==========', flush=True)
    print(
        f"Train -> R2: {train_metrics['r2']:.6f}, RMSE: {train_metrics['rmse']:.6f}, "
        f"MAPE: {train_metrics['mape']:.4f}%, MSE: {train_metrics['mse']:.8f}"
    )
    print(
        f"Test  -> R2: {test_metrics['r2']:.6f}, RMSE: {test_metrics['rmse']:.6f}, "
        f"MAPE: {test_metrics['mape']:.4f}%, MSE: {test_metrics['mse']:.8f}"
    )

    print('\n========== 分输出评估(Test) ==========', flush=True)
    for i, name in enumerate(target_cols):
        metric_i = evaluate_regression(y_test_real[:, i], test_pred[:, i])
        print(
            f"{name} -> R2: {metric_i['r2']:.6f}, RMSE: {metric_i['rmse']:.6f}, "
            f"MAPE: {metric_i['mape']:.4f}%, MSE: {metric_i['mse']:.8f}"
        )

    # 7) 保存模型、归一化器与配置
    model_path = OUTPUT_DIR / MODEL_FILE
    scaler_x_path = OUTPUT_DIR / SCALER_X_FILE
    scaler_y_path = OUTPUT_DIR / SCALER_Y_FILE
    config_path = OUTPUT_DIR / MODEL_CONFIG_FILE

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler_x, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)

    model_config = {
        'input_cols': input_cols,
        'target_cols': target_cols,
        'target_col': target_cols[0],
        'input_size': len(input_cols),
        'hidden_size': hidden_size,
        'output_size': output_size,
        'activation_hidden': ACTIVATION_HIDDEN,
        'activation_output': ACTIVATION_OUTPUT,
        'normalization': NORMALIZATION_METHOD,
        'train_algo': cfg.train_algo,
        'use_pso_init': cfg.use_pso_init,
        'seed': SEED,
        'output_dir': str(OUTPUT_DIR),
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    print('\n保存完成:')
    print(f'- {model_path}')
    print(f'- {scaler_x_path} / {scaler_y_path}')
    print(f'- {config_path}')
    print(f'- {OUTPUT_DIR / LOG_FILE}')
    print('训练结束。')


if __name__ == '__main__':
    main()
