#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// Auto-generated from export_to_c.py
// Network: BP (tansig + purelin)

#define INPUT_SIZE 3
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 2

// MinMaxScaler parameters (input): x_scaled = x * INPUT_SCALE + INPUT_MIN
const float INPUT_SCALE[] = {9.67000961f, 177.50944519f, 1.07382596f};
const float INPUT_MIN[] = {6.27823877f, -174.89712524f, -37.24162674f};

// MinMaxScaler parameters (output): y_scaled = y * OUTPUT_SCALE + OUTPUT_MIN
const float OUTPUT_SCALE[] = {0.23474178f, 2.38237047f};
const float OUTPUT_MIN[] = {-1.78873229f, -1.20845747f};

// Layer 1: fc1 (INPUT_SIZE -> HIDDEN_SIZE), activation=tansig
#define W1_ROWS HIDDEN_SIZE
#define W1_COLS INPUT_SIZE
const float W1[] = {0.54562229f, -0.15632460f, -2.04162431f, 2.37093496f, -0.22591770f, -1.68689930f, -1.24257839f, -0.51891953f, -1.66286111f};
const float B1[] = {-1.23767066f, 0.42943507f, -0.12247107f};

// Layer 2: fc2 (HIDDEN_SIZE -> OUTPUT_SIZE), activation=purelin
#define W2_ROWS OUTPUT_SIZE
#define W2_COLS HIDDEN_SIZE
const float W2[] = {0.58202904f, 0.01622156f, -0.99240577f, -2.11192155f, 1.75207436f, 0.21964034f};
const float B2[] = {0.17301093f, -1.05933177f};

// Metadata
// input_cols: 254,550,tem
// target_cols: cod,uv254

#endif // MODEL_DATA_H
