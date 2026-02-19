import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


# 工程根目录与统一输出目录
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class BPNet(nn.Module):
    # 与训练脚本一致的 BP 结构
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)


def float_array_to_string(arr, name):
    # 将 numpy 数组转换为 C 语言 float 数组字符串
    data_str = ', '.join([f'{x:.8f}f' for x in arr.flatten()])
    return f'const float {name}[] = {{{data_str}}};\n'


def main():
    # 1) 检查依赖文件
    required = [
        OUTPUT_DIR / 'water_quality_model.pth',
        OUTPUT_DIR / 'scaler_x.pkl',
        OUTPUT_DIR / 'scaler_y.pkl',
        OUTPUT_DIR / 'model_config.json',
    ]
    for path in required:
        if not path.exists():
            print(f'错误: 缺少文件 {path}，请先运行 new.py。')
            return

    with open(OUTPUT_DIR / 'model_config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # 兼容旧配置
    target_cols = cfg.get('target_cols')
    if not target_cols:
        target_cols = [cfg.get('target_col', 'target')]

    # 2) 加载模型与归一化参数
    model = BPNet(cfg['input_size'], cfg['hidden_size'], cfg['output_size'])
    model.load_state_dict(torch.load(OUTPUT_DIR / 'water_quality_model.pth'))
    model.eval()

    scaler_x = joblib.load(OUTPUT_DIR / 'scaler_x.pkl')
    scaler_y = joblib.load(OUTPUT_DIR / 'scaler_y.pkl')

    state_dict = model.state_dict()
    w1 = state_dict['fc1.weight'].numpy()  # [hidden, input]
    b1 = state_dict['fc1.bias'].numpy()    # [hidden]
    w2 = state_dict['fc2.weight'].numpy()  # [output, hidden]
    b2 = state_dict['fc2.bias'].numpy()    # [output]

    # 3) 拼接头文件内容
    header = '#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n'
    header += '// Auto-generated from export_to_c.py\n'
    header += '// Network: BP (tansig + purelin)\n\n'

    header += f'#define INPUT_SIZE {cfg["input_size"]}\n'
    header += f'#define HIDDEN_SIZE {cfg["hidden_size"]}\n'
    header += f'#define OUTPUT_SIZE {cfg["output_size"]}\n\n'

    header += '// MinMaxScaler parameters (input): x_scaled = x * INPUT_SCALE + INPUT_MIN\n'
    header += float_array_to_string(np.array(scaler_x.scale_, dtype=np.float32), 'INPUT_SCALE')
    header += float_array_to_string(np.array(scaler_x.min_, dtype=np.float32), 'INPUT_MIN')

    header += '\n// MinMaxScaler parameters (output): y_scaled = y * OUTPUT_SCALE + OUTPUT_MIN\n'
    header += float_array_to_string(np.array(scaler_y.scale_, dtype=np.float32), 'OUTPUT_SCALE')
    header += float_array_to_string(np.array(scaler_y.min_, dtype=np.float32), 'OUTPUT_MIN')

    header += '\n// Layer 1: fc1 (INPUT_SIZE -> HIDDEN_SIZE), activation=tansig\n'
    header += '#define W1_ROWS HIDDEN_SIZE\n#define W1_COLS INPUT_SIZE\n'
    header += float_array_to_string(w1, 'W1')
    header += float_array_to_string(b1, 'B1')

    header += '\n// Layer 2: fc2 (HIDDEN_SIZE -> OUTPUT_SIZE), activation=purelin\n'
    header += '#define W2_ROWS OUTPUT_SIZE\n#define W2_COLS HIDDEN_SIZE\n'
    header += float_array_to_string(w2, 'W2')
    header += float_array_to_string(b2, 'B2')

    header += '\n// Metadata\n'
    header += f'// input_cols: {",".join(cfg["input_cols"])}\n'
    header += f'// target_cols: {",".join(target_cols)}\n'

    header += '\n#endif // MODEL_DATA_H\n'

    # 4) 输出到 outputs/model_data.h
    model_data_path = OUTPUT_DIR / 'model_data.h'
    with open(model_data_path, 'w', encoding='utf-8') as f:
        f.write(header)

    print(f'导出完成: {model_data_path}')


if __name__ == '__main__':
    main()
