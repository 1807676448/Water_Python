# Water_Python — 水质神经网络训练与 STM32 部署

## 项目概述

Water_Python 是一套面向水质参数预测的**神经网络训练与嵌入式部署工具链**。项目通过 BP 神经网络建立从光谱/温度输入到水质指标输出的非线性映射模型，支持在 Python 环境完成训练与验证，并将模型参数导出为纯 C 头文件，部署到 STM32 等嵌入式平台进行离线推理。

### 核心能力

- **神经网络训练**：单隐层 BP 网络（`tansig` 激活 + 线性输出），支持 Adam/SGD/L-BFGS 三种优化算法
- **正则化与早停**：L2 权重衰减 + 验证损失监控 + 早停 + 学习率自适应调度，有效抑制小样本过拟合
- **PSO 权重初始化**：可选粒子群优化（Particle Swarm Optimization）预搜索更优初始权重，提升收敛速度与泛化能力
- **多输出预测**：同时预测多个目标水质参数（默认 COD + UV254）
- **归一化管线**：统一采用 `MinMaxScaler(-1, 1)`，训练与推理严格一致
- **C 代码导出**：一键将训练好的权重、偏置和归一化参数导出为 C 头文件
- **STM32 推理引擎**：配套纯 C 实现的推理库，与 PyTorch 输出结果逐位一致
- **图形化界面**：基于 Tkinter 的可视化工具箱，支持一键训练、导出、加载模型与交互式预测

### 默认场景

| 项目 | 说明 |
|------|------|
| 输入 | `254`（254nm 吸光度）、`550`（550nm 吸光度）、`tem`（温度） |
| 输出 | `cod`（化学需氧量）、`uv254`（UV254 值） |
| 网络结构 | 单隐层 BP：3 → 3 → 2（20参数，适配47条小样本） |
| 隐层激活 | `tanh`（tansig） |
| 输出激活 | 线性（purelin） |
| 归一化 | `MinMaxScaler(-1, 1)` |

---

## 项目结构

```text
Water_Python/
├── new.py                          # 训练脚本（核心入口）
├── use_model.py                    # 命令行交互式推理脚本
├── export_to_c.py                  # 模型参数导出为 C 头文件
├── ui_app.py                       # 图形化工具箱（训练/导出/加载/预测）
├── water_quality_ai.c              # STM32 推理引擎实现（推荐使用）
├── water_quality_ai.h              # STM32 推理引擎接口声明
├── main.c                          # 历史示例代码（三层 ReLU，不与当前导出链路兼容）
├── shuizhi.xlsx                    # 训练数据文件（默认名称）
├── 神经网络训练说明.md              # 中文训练与部署详细说明
├── __pycache__/                    # Python 字节码缓存
└── outputs/                        # 统一输出目录
    ├── water_quality_model.pth     # PyTorch 模型权重
    ├── scaler_x.pkl                # 输入 MinMaxScaler 归一化器
    ├── scaler_y.pkl                # 输出 MinMaxScaler 归一化器
    ├── model_config.json           # 模型结构与字段映射配置
    ├── model_data.h                # 导出给 C/STM32 的参数头文件
    ├── training_log.txt            # 训练日志（含指标）
    ├── ui_param_memory.json        # GUI 参数记忆
    ├── ui_train_config.json        # GUI 训练配置保存
    └── ui_train_config_test.json   # GUI 测试配置
```

---

## 环境要求与安装

### 系统要求

- **操作系统**：Windows / Linux / macOS
- **Python 版本**：3.9 及以上
- **STM32 编译环境**（仅嵌入式部署需要）：支持 C99 的 ARM GCC 工具链

### Python 依赖

| 库 | 用途 | 最低版本建议 |
|-----|------|-------------|
| `numpy` | 数值计算 | ≥1.20 |
| `pandas` | Excel 数据读取与处理 | ≥1.4 |
| `torch`（PyTorch） | 神经网络构建与训练 | ≥1.10 |
| `scikit-learn` | 数据划分、归一化、回归指标 | ≥1.0 |
| `joblib` | 归一化器序列化 | ≥1.1 |
| `openpyxl` | `.xlsx` 文件解析引擎 | ≥3.0 |

### 安装步骤

```bash
# 1. 克隆或进入项目目录
cd Water_Python

# 2. 安装 Python 依赖
pip install numpy pandas torch scikit-learn joblib openpyxl

# 3. 验证安装
python -c "import torch; import pandas; import sklearn; print('环境就绪')"
```

---

## 配置指南

### 训练参数宏定义（new.py）

所有可调整参数集中在 [new.py](file:///d:/STM32/Water_Python/new.py) 顶部，以模块级常量形式定义：

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `SEED` | `42` | 随机种子，影响数据划分、初始化与训练可复现性 |
| `DEFAULT_EXCEL_FILE` | `'shuizhi.xlsx'` | 训练数据文件名 |
| `DEFAULT_TEST_SIZE` | `0.2` | 测试集占比（20%） |
| `DEFAULT_SCENARIO` | `'custom'` | 训练场景：`'custom'` / `'temperature_compensation'` / `'turbidity_prediction'` |
| `DEFAULT_INPUT_COLS` | `('254', '550', 'tem')` | 输入特征列名（需与 Excel 列名一致） |
| `DEFAULT_TARGET_COLS` | `('cod', 'uv254')` | 输出目标列名（需与 Excel 列名一致） |
| `DEFAULT_HIDDEN_SIZE` | `8` | 隐层神经元数量 |
| `DEFAULT_LEARNING_RATE` | `0.05` | SGD 学习率 |
| `DEFAULT_MAX_EPOCHS` | `10000` | 最大训练轮数 |
| `DEFAULT_TARGET_MSE` | `1e-5` | 早停目标误差（达到后提前终止） |
| `DEFAULT_TRAIN_ALGO` | `'gd'` | 优化算法：`'gd'`（随机梯度下降）或 `'trainlm_like'`（L-BFGS） |
| `DEFAULT_USE_PSO_INIT` | `True` | 是否启用 PSO 预初始化权重 |
| `DEFAULT_PSO_PARTICLES` | `20` | PSO 粒子数量 |
| `DEFAULT_PSO_ITERS` | `60` | PSO 最大迭代次数 |
| `DEFAULT_PSO_W_MAX` | `0.9` | PSO 惯性权重（初始） |
| `DEFAULT_PSO_W_MIN` | `0.4` | PSO 惯性权重（终止） |
| `DEFAULT_PSO_C1` | `1.49445` | PSO 个体认知系数 |
| `DEFAULT_PSO_C2` | `1.49445` | PSO 社会认知系数 |

### 预置训练场景

通过 `DEFAULT_SCENARIO` 可切换以下三种场景：

| 场景 | 输入列 | 输出列 | 隐层大小 |
|------|--------|--------|----------|
| `temperature_compensation` | `tem`, `light_power`, `dark_current` | `cod` | 4 |
| `turbidity_prediction` | `scatter_voltage`, `transmit_voltage`, `ratio` | `turbidity` | 14 |
| `custom`（默认） | 由 `DEFAULT_INPUT_COLS` / `DEFAULT_TARGET_COLS` 指定 | `cod`, `uv254` | `DEFAULT_HIDDEN_SIZE` |

### 数据格式要求

训练数据文件（默认 `shuizhi.xlsx`）需满足以下条件：

- 格式：`.xlsx`（Excel 2007+）
- 第一行为列名，必须包含配置中的所有输入列和输出列
- 列名大小写和空格需与配置完全一致
- 数据区域无空行/空列，数值类型为数字

---

## 使用示例

### 工作流一：Python 端完整闭环

```bash
# 步骤 1：训练模型
python new.py

# 步骤 2：命令行验证预测
python use_model.py
# 交互输入示例：
# 请输入输入值(空格分隔，输入 q 退出): 1.23 0.56 25
# 预测结果 -> cod: 12.345678，uv254: 0.123456

# 步骤 3：导出 C 头文件
python export_to_c.py
```

### 工作流二：图形界面操作

```bash
python ui_app.py
```

GUI 提供四个核心功能：

1. **训练模型** — 调用 `new.py`，日志实时显示在界面底部
2. **导出参数** — 调用 `export_to_c.py`，生成 `outputs/model_data.h`
3. **加载模型** — 从 `outputs/` 读取模型，动态生成输入控件
4. **执行预测** — 在输入框中填写数值后点击预测，结果即时显示

### 工作流三：STM32 嵌入式部署

```bash
# 1. 在 Python 端完成训练和导出
python new.py
python export_to_c.py

# 2. 将以下三个文件复制到 STM32 工程
#    - water_quality_ai.c
#    - water_quality_ai.h
#    - outputs/model_data.h
```

STM32 端调用示例：

```c
#include "water_quality_ai.h"

int main(void) {
    float input[INPUT_SIZE] = {1.23f, 0.56f, 25.0f};
    float output[OUTPUT_SIZE] = {0};

    WaterQuality_Predict_Array(input, output);

    float cod_value   = output[0];
    float uv254_value = output[1];

    while (1) {
        // 业务逻辑
    }
}
```

---

## API 参考

### Python API

#### new.py — 训练模块

核心类和函数：

| 名称 | 类型 | 说明 |
|------|------|------|
| `TrainConfig` | 数据类（`@dataclass`） | 训练参数集中管理，字段对应上方配置表中的所有参数 |
| `BPNet` | `nn.Module` 子类 | 单隐层 BP 网络：`fc1(linear)` → `tanh` → `fc2(linear)` |
| `resolve_columns(df, cfg)` | 函数 | 根据场景配置解析输入/输出列名与隐层大小 |
| `flatten_params(model)` | 函数 | 将模型所有参数展平为一维 numpy 数组 |
| `assign_params(model, flat)` | 函数 | 将一维数组赋值回模型参数 |
| `pso_initialize(model, x, y, cfg)` | 函数 | PSO 权重预初始化，返回优化后的全局最优 MSE |
| `mape(y_true, y_pred)` | 函数 | 计算 MAPE（平均绝对百分比误差） |
| `evaluate_regression(y_true, y_pred)` | 函数 | 综合评估：返回 `mse`、`rmse`、`r2`、`mape` 字典 |

#### use_model.py — 推理模块

| 名称 | 类型 | 说明 |
|------|------|------|
| `load_model_system()` | 函数 | 加载模型、归一化器与配置；返回 `(model, scaler_x, scaler_y, cfg)` 四元组 |
| `predict_multi(model, sx, sy, values)` | 函数 | 单次预测：输入归一化 → 前向传播 → 输出反归一化 |
| `evaluate_regression(y_true, y_pred)` | 函数 | 回归指标评估（同 new.py） |
| `print_training_param_explanation(cfg)` | 函数 | 打印训练参数说明 |

#### export_to_c.py — 导出模块

| 名称 | 类型 | 说明 |
|------|------|------|
| `float_array_to_string(arr, name)` | 函数 | 将 numpy 数组转换为 C 语言 `const float[]` 声明字符串 |
| `main()` | 函数 | 主流程：加载模型参数 → 生成 `outputs/model_data.h` |

#### ui_app.py — GUI 模块

| 名称 | 类型 | 说明 |
|------|------|------|
| `WaterQualityUI` | 类（`tk.Tk` 包装） | GUI 主窗口，封装所有交互逻辑 |
| `_build_layout()` | 方法 | 构建界面布局（按钮、输入区、日志区） |
| `run_training()` | 方法 | 执行 `new.py` 训练 |
| `run_export()` | 方法 | 执行 `export_to_c.py` 导出 |
| `load_model()` | 方法 | 加载模型并动态渲染输入字段 |
| `predict()` | 方法 | 获取输入值并执行推理 |
| `_render_input_fields(fields)` | 方法 | 根据模型输入列动态创建 Entry 控件 |
| `_append_log(text)` | 方法 | 向日志框追加文本 |
| `_run_script(script_name)` | 方法 | 以子进程方式运行外部 Python 脚本并捕获输出 |

### C API（STM32 推理引擎）

接口定义位于 [water_quality_ai.h](file:///d:/STM32/Water_Python/water_quality_ai.h)，实现位于 [water_quality_ai.c](file:///d:/STM32/Water_Python/water_quality_ai.c)。

#### WaterQuality_Predict_Array

```c
void WaterQuality_Predict_Array(const float* input, float* out_values);
```

**推荐使用的通用接口**，执行完整的神经网络前向传播：

1. 输入 MinMax 归一化：`x_scaled = x * INPUT_SCALE + INPUT_MIN`
2. 全连接层 1 + tansig 激活
3. 全连接层 2 + 线性输出
4. 输出反归一化：`y = (y_scaled - OUTPUT_MIN) / OUTPUT_SCALE`

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `const float*` | 输入向量，长度必须为 `INPUT_SIZE` |
| `out_values` | `float*` | 输出向量（调用者分配），长度必须为 `OUTPUT_SIZE` |

#### WaterQuality_Predict3

```c
void WaterQuality_Predict3(float in_0, float in_1, float in_2, float* out_values);
```

**三输入快捷接口**，当模型输入维度为 3 时的简化调用形式。

| 参数 | 类型 | 说明 |
|------|------|------|
| `in_0`, `in_1`, `in_2` | `float` | 三个输入值（对应 254、550、tem） |
| `out_values` | `float*` | 输出向量（调用者分配） |

#### WaterQuality_Predict

```c
void WaterQuality_Predict(float in_254, float in_550, float in_tem,
                          float* out_cod, float* out_uv254);
```

**旧版兼容接口**，以命名参数形式传递输入，以独立指针接收双输出。当 `OUTPUT_SIZE < 2` 时，`out_uv254` 将被置为 0。

#### 内部辅助函数（static）

| 函数 | 说明 |
|------|------|
| `float_tansig(data, size)` | 对数组逐元素应用 `tanhf()` 激活函数 |
| `layer_dense(input, weights, bias, output, rows, cols)` | 全连接层计算：`y = W·x + b` |

#### model_data.h 宏定义

| 宏 | 类型 | 说明 |
|-----|------|------|
| `INPUT_SIZE` | `#define` | 输入特征维度 |
| `HIDDEN_SIZE` | `#define` | 隐层神经元数量 |
| `OUTPUT_SIZE` | `#define` | 输出目标维度 |
| `INPUT_SCALE[N]` | `const float[]` | 输入归一化 scale 参数 |
| `INPUT_MIN[N]` | `const float[]` | 输入归一化 min 参数 |
| `OUTPUT_SCALE[N]` | `const float[]` | 输出归一化 scale 参数 |
| `OUTPUT_MIN[N]` | `const float[]` | 输出归一化 min 参数 |
| `W1[HIDDEN_SIZE][INPUT_SIZE]` | `const float[]` | 第一层权重（展平为 1D） |
| `B1[HIDDEN_SIZE]` | `const float[]` | 第一层偏置 |
| `W2[OUTPUT_SIZE][HIDDEN_SIZE]` | `const float[]` | 第二层权重（展平为 1D） |
| `B2[OUTPUT_SIZE]` | `const float[]` | 第二层偏置 |

---

## 架构设计

### 数据流

```text
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  shuizhi.xlsx │ ──▶ │   new.py     │ ──▶ │  outputs/*.pth   │
│  (训练数据)   │     │  (训练+评估)  │     │  outputs/*.pkl   │
└─────────────┘     └──────────────┘     │  outputs/*.json  │
                                          └────────┬─────────┘
                                                   │
                          ┌────────────────────────┼────────────────────────┐
                          │                        │                        │
                          ▼                        ▼                        ▼
                   ┌─────────────┐        ┌──────────────┐        ┌─────────────────┐
                   │ use_model.py │        │export_to_c.py│        │   ui_app.py      │
                   │ (命令行推理)  │        │ (C 头文件导出)│        │ (图形化工具箱)    │
                   └─────────────┘        └──────┬───────┘        └─────────────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │ model_data.h     │
                                        │ (C 参数头文件)    │
                                        └────────┬────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────────┐
                                        │ water_quality_ai.c   │
                                        │ (STM32 推理引擎)     │
                                        └─────────────────────┘
```

### 网络前向传播公式

```
z₁ = W₁ · x + b₁          (全连接层 1)
a₁ = tanh(z₁)             (tansig 激活)
z₂ = W₂ · a₁ + b₂         (全连接层 2)
ŷ  = z₂                   (purelin 线性输出)
```

### 归一化与反归一化

```text
归一化（输入/输出）:
  x_scaled = (x - data_min) / (data_max - data_min) * (1 - (-1)) + (-1)
           = x * INPUT_SCALE + INPUT_MIN
  其中 INPUT_SCALE = 2 / (data_max - data_min)，INPUT_MIN = -1 - data_min * INPUT_SCALE

反归一化（输出）:
  y = (y_scaled - OUTPUT_MIN) / OUTPUT_SCALE
```

### PSO 初始化算法

粒子群优化在训练前执行，用于搜索更优的初始权重：

1. 随机初始化 `N` 个粒子（`pso_particles` 个），每个粒子代表一组完整的网络权重
2. 迭代 `pso_iters` 轮：
   - 对每个粒子，将权重赋给网络，计算训练集 MSE 作为适应度
   - 更新个体最优位置 `pbest` 和全局最优位置 `gbest`
   - 按标准 PSO 公式更新速度和位置，惯性权重 `w` 随迭代次数二次衰减
3. 将全局最优位置的权重赋给网络，继续 BP 训练

PSO 参数建议：
- 粒子数 `10~30`，迭代次数 `20~80`
- 惯性权重从 `0.9` 线性/二次衰减至 `0.4`
- `c1 = c2 = 1.49445`（经典配置）或 `c1 = c2 = 2.0`

---

## 输出文件说明

| 文件 | 生成时机 | 格式 | 说明 |
|------|----------|------|------|
| `water_quality_model.pth` | `new.py` 训练完成 | PyTorch state_dict | 网络权重与偏置 |
| `scaler_x.pkl` | `new.py` 训练完成 | joblib 序列化 | 输入 MinMaxScaler |
| `scaler_y.pkl` | `new.py` 训练完成 | joblib 序列化 | 输出 MinMaxScaler |
| `model_config.json` | `new.py` 训练完成 | JSON | 模型元信息（列名、网络维度等） |
| `training_log.txt` | `new.py` 全运行过程 | 文本 | 完整训练日志（参数、PSO 进度、每轮误差、最终指标） |
| `model_data.h` | `export_to_c.py` 执行 | C 头文件 | 导出给 STM32 的权重/归一化参数 |
| `ui_param_memory.json` | `ui_app.py` 参数记忆 | JSON | GUI 参数预设记忆 |
| `ui_train_config.json` | `ui_app.py` 配置保存 | JSON | GUI 训练参数持久化 |

---

## 评估指标说明

训练完成后输出的四项指标：

| 指标 | 全称 | 范围 | 评价标准 |
|------|------|------|----------|
| **MSE** | Mean Squared Error（均方误差） | `[0, +∞)` | 越小越好，对大误差敏感 |
| **RMSE** | Root Mean Squared Error（均方根误差） | `[0, +∞)` | 越小越好，量纲与目标值一致 |
| **R²** | Coefficient of Determination（决定系数） | `(-∞, 1]` | 越接近 1 越好，负值表示模型劣于均值预测 |
| **MAPE** | Mean Absolute Percentage Error（平均绝对百分比误差） | `[0, +∞)` | 越小越好，以百分比形式衡量相对误差 |

训练日志会输出总体评估与各目标指标分列的评估结果，便于针对性调优。

---

## 故障排除

### Q1：训练时提示 "数据列缺失"

**原因**：Excel 列名与 `new.py` 中配置的输入/输出列名不匹配。

**解决**：
1. 检查 Excel 第一行的列名（大小写、空格完全一致）
2. 确保 Excel 中同时存在所有输入列和目标列
3. 可临时修改 `new.py` 顶部的 `DEFAULT_INPUT_COLS` / `DEFAULT_TARGET_COLS`

### Q2：推理时提示 "未找到 xxx 文件，请先运行 new.py"

**原因**：`outputs/` 目录下缺少训练产物。

**解决**：
```bash
python new.py   # 先完整运行训练
```
确认 `outputs/` 中存在以下文件：
- `water_quality_model.pth`
- `scaler_x.pkl`
- `scaler_y.pkl`
- `model_config.json`

### Q3：导出时报错或生成的 model_data.h 为空

**原因**：导出的依赖文件不完整。

**解决**：按顺序执行：
```bash
python new.py           # 确保训练完成
python export_to_c.py   # 再导出
```

### Q4：STM32 推理结果与 Python 端不一致

**常见原因及排查顺序**：

1. **参数版本不一致** — 每次重新训练后都必须重新导出 `model_data.h`，确保 STM32 使用的是与 Python 训练同一批参数
2. **使用了错误的 C 文件** — 必须使用 `water_quality_ai.c`，不要使用 `main.c`（后者是旧版三层 ReLU 实现，与当前导出链路不兼容）
3. **编译器浮点配置** — 检查 STM32 工程的 FPU 设置：
   - 是否启用硬件浮点单元（`-mfloat-abi=hard -mfpu=fpv4-sp-d16` 等）
   - 若使用软件浮点（`-mfloat-abi=soft`），精度会有微小差异
4. **优化级别** — 高优化级别（`-O2`/`-O3`）可能改变浮点运算顺序，影响末位精度；可尝试 `-O1` 或关闭 `-ffast-math` 对比
5. **输入值预处理** — 确保 C 侧输入值与 Python 侧完全一致（包括数据类型和精度）

### Q5：PSO 初始化耗时长或效果不佳

**调整建议**：
- 减少 `DEFAULT_PSO_PARTICLES`（如改为 10~15）
- 减少 `DEFAULT_PSO_ITERS`（如改为 30~40）
- 若数据量小且问题简单，可设置 `DEFAULT_USE_PSO_INIT = False`，直接使用随机初始化

### Q6：模型过拟合（训练集效果好但测试集差）

**调整建议**：
- 减小 `DEFAULT_HIDDEN_SIZE`（如从 8 改为 4~6）
- 增加 `DEFAULT_TEST_SIZE`（如从 0.2 改为 0.3）
- 提高 `DEFAULT_TARGET_MSE`（如从 `1e-5` 改为 `1e-4`），避免过度训练
- 尝试关闭 PSO 初始化，使用更简单的初始化策略

### Q7：模型欠拟合（训练和测试效果都不好）

**调整建议**：
- 增大 `DEFAULT_HIDDEN_SIZE`（如从 8 改为 12~16）
- 增大 `DEFAULT_MAX_EPOCHS`（如从 10000 改为 20000）
- 降低 `DEFAULT_TARGET_MSE`（如从 `1e-5` 改为 `1e-6`）
- 调整学习率（尝试 `0.01` 或 `0.1`）
- 开启 PSO 初始化帮助找到更好的初始点

---

## 贡献指南

### 开发环境搭建

```bash
# 1. 克隆仓库
git clone <repository-url>
cd Water_Python

# 2. 创建虚拟环境（推荐）
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. 安装依赖
pip install numpy pandas torch scikit-learn joblib openpyxl

# 4. 运行现有测试流程验证环境
python new.py
python use_model.py
```

### 代码规范

- **Python 代码**：遵循 PEP 8 风格，使用 4 空格缩进
- **C 代码**：遵循项目现有风格（K&R 风格括号，`static` 修饰内部函数）
- **命名约定**：
  - Python 函数：`snake_case`
  - Python 类：`PascalCase`
  - Python 常量：`UPPER_SNAKE_CASE`
  - C 函数：`PascalCase`（公开接口）/ `snake_case`（static 内部函数）
- **注意**：本项目所有 Python 代码已去除注释，如需添加注释请保持与现有代码风格一致（中文注释，简洁明了）

### 目录规范

- 所有训练产物写入 `outputs/` 目录，不要散落在项目根目录
- 新增 Python 脚本使用 `Path(__file__).resolve().parent` 定位项目根目录
- C 代码中 `#include "model_data.h"` 需确保 Include Path 正确配置

### 提交流程

1. 确保修改后 `python new.py` 能完整运行并生成所有 `outputs/` 产物
2. 确保 `python use_model.py` 能正常加载模型并预测
3. 确保 `python export_to_c.py` 能正常生成 C 头文件
4. 检查输出日志中的评估指标是否在合理范围内
5. 提交信息请使用中文，清晰描述变更内容

### 扩展方向建议

- **新增网络结构**：在 `BPNet` 基础上扩展更深或更宽的网络（需同步修改 C 端推理引擎）
- **新增归一化方式**：当前仅支持 `MinMaxScaler(-1,1)`，可扩展 `StandardScaler` 等（需同步修改导出脚本和 C 端）
- **新增激活函数**：如在隐层添加 `ReLU`、`sigmoid` 选项（需在 C 端添加对应实现）
- **新增训练算法**：如 Adam、RMSprop 等优化器
- **增强 GUI**：添加训练曲线可视化、参数调优面板、模型对比等功能

### 重要注意事项

- **不要直接使用 `main.c`**：该文件为历史遗留的三层 ReLU 网络实现，与当前导出链路不兼容。STM32 部署请使用 `water_quality_ai.c` + `water_quality_ai.h` + `outputs/model_data.h`
- **训练后务必重新导出**：每次重新训练都会更新模型参数，必须重新运行 `export_to_c.py` 以保持 Python 侧和 C 侧参数同步
- **固定随机种子**：如需结果可复现，保持 `SEED = 42` 不变，并确保数据文件未被修改

---

## 相关资源

- 训练与部署中文详细说明：[神经网络训练说明.md](file:///d:/STM32/Water_Python/神经网络训练说明.md)
- 输出目录：[outputs/](file:///d:/STM32/Water_Python/outputs/)
- STM32 推理引擎（推荐）：[water_quality_ai.c](file:///d:/STM32/Water_Python/water_quality_ai.c) / [water_quality_ai.h](file:///d:/STM32/Water_Python/water_quality_ai.h)
- 历史示例（不推荐用于当前版本）：[main.c](file:///d:/STM32/Water_Python/main.c)
