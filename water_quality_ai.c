#include "water_quality_ai.h"
#include <math.h>

// -------------------------------------------------------------------------
// 包含权重文件
// 注意：请确保运行 export_to_c.py 后生成的 model_data.h 文件在项目的 Include Path 中
// -------------------------------------------------------------------------
#include "model_data.h"

// ------------------------------------------------------------
// 内部辅助函数 (static 修饰，限制在当前文件内可见，避免命名冲突)
// ------------------------------------------------------------

/**
 * @brief tansig 激活函数（双曲正切）
 * @param data 数据数组指针
 * @param size 数组长度
 */
static void float_tansig(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = tanhf(data[i]);
    }
}

/**
 * @brief 全连接层 (Dense/Linear) 计算: y = W * x + b
 * @param input   输入向量 [cols]
 * @param weights 权重矩阵 [rows * cols] (展平的一维数组)
 * @param bias    偏置向量 [rows]
 * @param output  输出向量 [rows]
 * @param rows    输出维度
 * @param cols    输入维度
 */
static void layer_dense(const float* input, const float* weights, const float* bias, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        // 矩阵乘法: 累加 input[j] * weights[i, j]
        for (int j = 0; j < cols; j++) {
            // 注意: 这里的索引映射与 export_to_c.py 的导出顺序一致
            // PyTorch Linear权重默认是 [Out, In] (Row-Major flattening)
            sum += input[j] * weights[i * cols + j];
        }
        output[i] = sum + bias[i];
    }
}

// ------------------------------------------------------------
// 核心接口实现
// ------------------------------------------------------------

void WaterQuality_Predict_Array(const float* input, float* out_values) {
    if (input == 0 || out_values == 0) {
        return;
    }

    // 0. 中间缓冲
    float input_scaled[INPUT_SIZE];
    float hidden[W1_ROWS];
    float output_scaled[W2_ROWS];

    // 1. 输入归一化 (MinMaxScaler)
    // x_scaled = x * INPUT_SCALE + INPUT_MIN
    for (int i = 0; i < INPUT_SIZE; i++) {
        input_scaled[i] = input[i] * INPUT_SCALE[i] + INPUT_MIN[i];
    }

    // 2. 隐含层: FC1 + tansig
    layer_dense(input_scaled, W1, B1, hidden, W1_ROWS, W1_COLS);
    float_tansig(hidden, W1_ROWS);

    // 3. 输出层: FC2 + purelin
    layer_dense(hidden, W2, B2, output_scaled, W2_ROWS, W2_COLS);

    // 4. 输出反归一化 (MinMaxScaler)
    // y = (y_scaled - OUTPUT_MIN) / OUTPUT_SCALE
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (OUTPUT_SCALE[i] == 0.0f) {
            out_values[i] = 0.0f;
        } else {
            out_values[i] = (output_scaled[i] - OUTPUT_MIN[i]) / OUTPUT_SCALE[i];
        }
    }
}

void WaterQuality_Predict3(float in_0, float in_1, float in_2, float* out_values) {
    float input[INPUT_SIZE] = {0.0f};
    if (INPUT_SIZE > 0) input[0] = in_0;
    if (INPUT_SIZE > 1) input[1] = in_1;
    if (INPUT_SIZE > 2) input[2] = in_2;
    WaterQuality_Predict_Array(input, out_values);
}

void WaterQuality_Predict(float in_254, float in_550, float in_tem, float* out_cod, float* out_uv254) {
    float outputs[OUTPUT_SIZE];
    WaterQuality_Predict3(in_254, in_550, in_tem, outputs);

    if (out_cod != 0) {
        *out_cod = (OUTPUT_SIZE > 0) ? outputs[0] : 0.0f;
    }
    if (out_uv254 != 0) {
        *out_uv254 = (OUTPUT_SIZE > 1) ? outputs[1] : 0.0f;
    }
}
