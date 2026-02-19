#ifndef WATER_QUALITY_AI_H
#define WATER_QUALITY_AI_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * @brief 水质预测 AI 模型接口
 * 
 * 该模块实现了基于 BP 神经网络（单隐层）的推理逻辑：
 * INPUT -> FC1 -> tansig -> FC2 -> purelin
 * 权重数据在 "model_data.h" 中定义。
 */

/**
 * @brief 核心预测函数（推荐）
 * 此函数执行完整的神经网络前向传播过程：
 * 1. 输入归一化 (MinMaxScaler): x_scaled = x * scale + min
 * 2. 神经网络推理 (FC -> tansig -> FC)
 * 3. 输出反归一化 (MinMaxScaler): y = (y_scaled - min) / scale
 * 
 * @param input      输入向量，长度为 INPUT_SIZE
 * @param out_values 输出向量，长度为 OUTPUT_SIZE
 */
void WaterQuality_Predict_Array(const float* input, float* out_values);

/**
 * @brief 3输入快捷函数（当模型输入为3时推荐）
 * @param in_0 输入0
 * @param in_1 输入1
 * @param in_2 输入2
 * @param out_values 输出向量，长度为 OUTPUT_SIZE
 */
void WaterQuality_Predict3(float in_0, float in_1, float in_2, float* out_values);

/**
 * @brief 兼容旧接口（旧版双输出）
 * @note 若 OUTPUT_SIZE 为1，则 out_uv254 置0。
 */
void WaterQuality_Predict(float in_254, float in_550, float in_tem, float* out_cod, float* out_uv254);

#ifdef __cplusplus
}
#endif

#endif // WATER_QUALITY_AI_H
