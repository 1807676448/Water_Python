#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// Auto-generated from export_to_c.py
// Network: BP (tansig + purelin)

#define INPUT_SIZE 2
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 2

// MinMaxScaler parameters (input): x_scaled = x * INPUT_SCALE + INPUT_MIN
const float INPUT_SCALE[] = {3.92063808f, 125.77849579f};
const float INPUT_MIN[] = {4.03447199f, -123.66094971f};

// MinMaxScaler parameters (output): y_scaled = y * OUTPUT_SCALE + OUTPUT_MIN
const float OUTPUT_SCALE[] = {0.13586958f, 5.12557697f};
const float OUTPUT_MIN[] = {-1.50135875f, -1.49359310f};

// Layer 1: fc1 (INPUT_SIZE -> HIDDEN_SIZE), activation=tansig
#define W1_ROWS HIDDEN_SIZE
#define W1_COLS INPUT_SIZE
const float W1[] = {0.51139307f, 0.05800919f, 0.51196080f, 0.05819561f, 0.97149688f, 0.01196654f};
const float B1[] = {-0.07563849f, -0.07600144f, -0.50365704f};

// Layer 2: fc2 (HIDDEN_SIZE -> OUTPUT_SIZE), activation=purelin
#define W2_ROWS OUTPUT_SIZE
#define W2_COLS HIDDEN_SIZE
const float W2[] = {0.40414545f, 0.40470648f, 0.77396876f, 0.40435702f, 0.40485650f, 0.77409923f};
const float B2[] = {0.14204815f, 0.14176588f};

// Metadata
// input_cols: comp_254,comp_550
// target_cols: COD_lab,UV254_lab

#endif // MODEL_DATA_H
