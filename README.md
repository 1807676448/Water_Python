# Water_Python（水质神经网络训练与 STM32 部署）

本项目用于训练水质预测神经网络模型，并将模型参数导出到 C 头文件，方便在 STM32 侧进行离线推理。

当前默认场景：
- 输入：`254`、`550`、`tem`
- 输出：`cod`、`uv254`
- 网络结构：单隐层 BP（`tanh/tansig` + 线性输出）
- 归一化：`MinMaxScaler(-1, 1)`

---

## 1. 项目结构

```text
Water_Python/
├─ new.py                      # 训练脚本（核心）
├─ use_model.py                # 命令行推理脚本
├─ export_to_c.py              # 导出 C 头文件参数
├─ ui_app.py                   # 图形化工具（训练/导出/加载/预测）
├─ water_quality_ai.c          # STM32 推理实现（推荐）
├─ water_quality_ai.h          # STM32 推理接口声明
├─ main.c                      # 旧版/历史示例（与当前导出链路不一致）
├─ 神经网络训练说明.md          # 训练与部署说明（中文）
├─ outputs/
│  ├─ model_config.json
│  ├─ model_data.h
│  ├─ training_log.txt
│  └─ water_quality_model.pth
└─ ...
```

> 说明：`outputs/` 中的 `scaler_x.pkl`、`scaler_y.pkl` 会在训练后自动生成，供推理与导出使用。

---

## 2. 环境要求

- 操作系统：Windows（当前工程环境）
- Python：建议 3.9+
- 依赖库：
  - `numpy`
  - `pandas`
  - `torch`
  - `scikit-learn`
  - `joblib`
  - `openpyxl`（读取 `.xlsx`）

安装依赖（在项目根目录）：

```bash
pip install numpy pandas torch scikit-learn joblib openpyxl
```

---

## 3. 一分钟快速开始

### 第 1 步：准备训练数据

默认训练文件名为 `shuizhi.xlsx`（可在 `new.py` 中修改 `DEFAULT_EXCEL_FILE`）。

默认需要包含列：
- 输入列：`254`、`550`、`tem`
- 目标列：`cod`、`uv254`

### 第 2 步：训练模型

```bash
python new.py
```

训练完成后，会在 `outputs/` 生成（或更新）：
- `water_quality_model.pth`
- `scaler_x.pkl`
- `scaler_y.pkl`
- `model_config.json`
- `training_log.txt`

### 第 3 步：命令行预测

```bash
python use_model.py
```

输入格式（示例）：

```text
1.23 0.56 25
```

输入 `q` 退出。

### 第 4 步：导出到 C 头文件

```bash
python export_to_c.py
```

将生成：
- `outputs/model_data.h`

---

## 4. 脚本说明

### `new.py`（训练）

主要流程：
1. 读取训练配置（`TrainConfig`）
2. 加载 Excel 并校验输入/输出列
3. 划分训练集/测试集并进行归一化
4. 构建 BP 网络
5. 可选 PSO 初始化（`use_pso_init=True`）
6. 执行训练（`gd` 或 `trainlm_like`）
7. 计算并输出回归指标（`MSE/RMSE/R2/MAPE`）
8. 保存模型与配置到 `outputs/`

可在脚本顶部修改的常用参数：
- `DEFAULT_INPUT_COLS`
- `DEFAULT_TARGET_COLS`
- `DEFAULT_HIDDEN_SIZE`
- `DEFAULT_LEARNING_RATE`
- `DEFAULT_MAX_EPOCHS`
- `DEFAULT_TARGET_MSE`
- `DEFAULT_USE_PSO_INIT`

### `use_model.py`（命令行推理）

功能：
- 加载 `outputs/` 中模型、归一化器、配置
- 接收一组输入并输出多目标预测值
- 与训练配置自动对齐（含兼容旧配置兜底）

### `export_to_c.py`（参数导出）

功能：
- 读取训练好的 `.pth` 与 `scaler`
- 提取全连接层权重/偏置
- 生成 `outputs/model_data.h`

`model_data.h` 包含：
- 网络维度宏：`INPUT_SIZE/HIDDEN_SIZE/OUTPUT_SIZE`
- 归一化参数数组：`INPUT_SCALE/INPUT_MIN`、`OUTPUT_SCALE/OUTPUT_MIN`
- 权重与偏置数组：`W1/B1`、`W2/B2`

### `ui_app.py`（图形化界面）

提供三个核心按钮：
1. 训练模型（调用 `new.py`）
2. 导出参数（调用 `export_to_c.py`）
3. 加载模型（从 `outputs/`）

加载后会动态生成输入框，支持直接预测并显示输出字段结果。

---

## 5. STM32 集成说明

### 5.1 推荐文件组合

在 STM32 工程中使用以下 3 个文件：
- `water_quality_ai.c`
- `water_quality_ai.h`
- `outputs/model_data.h`（由导出脚本生成）

> `main.c` 当前为历史示例，采用的是旧网络定义（例如 ReLU/三层结构等），与当前导出的 `model_data.h` 链路不一致，不建议直接用于现版本。

### 5.2 推理接口

在 `water_quality_ai.h` 中提供：

- `WaterQuality_Predict_Array(const float* input, float* out_values)`
  - 通用接口，推荐使用
  - `input` 长度为 `INPUT_SIZE`
  - `out_values` 长度为 `OUTPUT_SIZE`

- `WaterQuality_Predict3(float in_0, float in_1, float in_2, float* out_values)`
  - 3 输入快捷接口

- `WaterQuality_Predict(float in_254, float in_550, float in_tem, float* out_cod, float* out_uv254)`
  - 兼容旧调用风格

### 5.3 调用示例

```c
float in[INPUT_SIZE] = {1.23f, 0.56f, 25.0f};
float out[OUTPUT_SIZE] = {0};

WaterQuality_Predict_Array(in, out);

float cod = (OUTPUT_SIZE > 0) ? out[0] : 0.0f;
float uv254 = (OUTPUT_SIZE > 1) ? out[1] : 0.0f;
```

---

## 6. 典型工作流

### Python 侧闭环

1. `python new.py`
2. `python use_model.py`（验证预测）
3. `python export_to_c.py`

### STM32 侧闭环

1. 从 Python 端导出最新 `outputs/model_data.h`
2. 替换 STM32 工程中的模型参数头文件
3. 使用 `water_quality_ai.c/.h` 调用预测接口
4. 下载运行并比对结果

---

## 7. 常见问题（FAQ）

### Q1：提示缺少列 / 数据列不匹配

检查 Excel 是否包含 `new.py` 配置中的输入/输出列名（大小写和空格都要一致）。

### Q2：提示找不到模型文件

先执行 `python new.py`，确认 `outputs/` 下已生成：
- `water_quality_model.pth`
- `scaler_x.pkl`
- `scaler_y.pkl`
- `model_config.json`

### Q3：导出时报错

`export_to_c.py` 依赖训练结果文件，若缺失任意一个会失败。请先完整跑通训练。

### Q4：STM32 推理结果和 Python 不一致

优先排查：
1. 是否使用同一份最新 `model_data.h`
2. 是否使用 `water_quality_ai.c`（而不是旧版 `main.c` 示例链路）
3. 编译器浮点配置是否正确（FPU/软浮点设置）

---

## 8. 建议与注意事项

- 每次重新训练后都应重新执行导出，保持 C 侧参数同步。
- 尽量固定随机种子和数据划分方式，便于结果复现。
- 模型上线前建议在独立测试集上记录误差指标（尤其是 `RMSE` 与 `R2`）。

---

## 9. 相关文件

- 训练与部署详细说明：`神经网络训练说明.md`
- 输出目录（模型与配置）：`outputs/`
