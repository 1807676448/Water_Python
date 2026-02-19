#include <stdio.h>
#include <math.h>

// 包含由 export_to_c.py 生成的权重数据
#include "model_data.h"

// ------------------------------------------------------------
// 简单的神经网络推理库 (纯C实现)
// ------------------------------------------------------------

// ReLU 激活函数: x > 0 ? x : 0
void relu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0) {
            data[i] = 0;
        }
    }
}

// 矩阵向量乘法 + 偏置 add: y = W * x + b
// W 维度: [rows x cols], x 维度: [cols], b 维度: [rows], y 维度: [rows]
// 注意：model_data.h 中的权重数组通常是展平的 1D 数组
void dense(const float* input, const float* weights, const float* bias, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            // PyTorch Linear 权重默认布局通常是 [Out, In]，即行优先
            // weights[i * cols + j] 对应第 i 个输出神经元连接到第 j 个输入的权重
            sum += input[j] * weights[i * cols + j];
        }
        output[i] = sum + bias[i];
    }
}

// ------------------------------------------------------------
// 主推理函数
// ------------------------------------------------------------
// 输入: input[3] (254, 550, tem)
// 输出: output[2] (cod, uv254)
void predict_water_quality(float in_254, float in_550, float in_tem, float* out_cod, float* out_uv254) {
    // 0. 定义临时缓冲区 (根据网络最大隐藏层大小设置)
    float layer1_out[W1_ROWS]; // 32
    float layer2_out[W2_ROWS]; // 16
    float layer3_out[W3_ROWS]; // 2 (Final output raw)

    // 1. 输入数据准备与标准化 (StandardScaler)
    // x_scaled = (x - mean) / scale
    float input[3];
    input[0] = (in_254 - INPUT_MEAN[0]) / INPUT_SCALE[0];
    input[1] = (in_550 - INPUT_MEAN[1]) / INPUT_SCALE[1];
    input[2] = (in_tem - INPUT_MEAN[2]) / INPUT_SCALE[2];

    // 2. 第一层 FC (3 -> 32) + ReLU
    dense(input, W1, B1, layer1_out, W1_ROWS, W1_COLS);
    relu(layer1_out, W1_ROWS);

    // 3. 第二层 FC (32 -> 16) + ReLU
    dense(layer1_out, W2, B2, layer2_out, W2_ROWS, W2_COLS);
    relu(layer2_out, W2_ROWS);

    // 4. 第三层 FC (16 -> 2) (输出层通常没有激活函数)
    dense(layer2_out, W3, B3, layer3_out, W3_ROWS, W3_COLS);

    // 5. 输出反标准化 (StandardScaler 反向)
    // y_real = y_pred * scale + mean
    *out_cod   = layer3_out[0] * OUTPUT_SCALE[0] + OUTPUT_MEAN[0];
    *out_uv254 = layer3_out[1] * OUTPUT_SCALE[1] + OUTPUT_MEAN[1];
}

// ------------------------------------------------------------
// 演示用的 main 函数
// (在单片机中，你只需要复制上面的函数，把 main 换成你的业务逻辑)
// ------------------------------------------------------------
int main() {
    float val_254 = 1.5f;
    float val_550 = 0.2f;
    float val_tem = 25.0f;
    float res_cod, res_uv;

    printf("Start Prediction...\n");
    printf("Input: 254=%.2f, 550=%.2f, tem=%.0f\n", val_254, val_550, val_tem);

    predict_water_quality(val_254, val_550, val_tem, &res_cod, &res_uv);

    printf("Result:\n");
    printf("  COD:   %.2f\n", res_cod);
    printf("  UV254: %.4f\n", res_uv);

    return 0;
}
