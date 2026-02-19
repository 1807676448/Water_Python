#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// Auto-generated from export_to_c.py
// Network: BP (tansig + purelin)

#define INPUT_SIZE 3
#define HIDDEN_SIZE 8
#define OUTPUT_SIZE 2

// MinMaxScaler parameters (input): x_scaled = x * INPUT_SCALE + INPUT_MIN
const float INPUT_SCALE[] = {62.52345657f, 18.65775871f, 0.18529329f};
const float INPUT_MIN[] = {-21.43610382f, -9.53484249f, -5.98670578f};

// MinMaxScaler parameters (output): y_scaled = y * OUTPUT_SCALE + OUTPUT_MIN
const float OUTPUT_SCALE[] = {1.01010108f, 38.16793823f};
const float OUTPUT_MIN[] = {-3.53535366f, -3.48091578f};

// Layer 1: fc1 (INPUT_SIZE -> HIDDEN_SIZE), activation=tansig
#define W1_ROWS HIDDEN_SIZE
#define W1_COLS INPUT_SIZE
const float W1[] = {-1.09875917f, -0.58433014f, 0.06543572f, 1.98008466f, -1.13598502f, -0.62284261f, -0.55906373f, -1.82242751f, 1.02080703f, -0.21841198f, -0.98724955f, 1.61540413f, -0.07908110f, -0.55236799f, -1.39121294f, -0.08429763f, 1.32627153f, -1.46390355f, -0.69519907f, -0.27432597f, -1.40326405f, 2.60964465f, -0.63328457f, -0.47223341f};
const float B1[] = {-2.22142577f, 0.25479501f, 0.14244004f, -1.06799376f, 0.79965502f, 0.83874685f, 1.81121325f, -1.23204982f};

// Layer 2: fc2 (HIDDEN_SIZE -> OUTPUT_SIZE), activation=purelin
#define W2_ROWS OUTPUT_SIZE
#define W2_COLS HIDDEN_SIZE
const float W2[] = {-0.75762165f, 1.03198719f, -0.03043784f, -0.51571405f, -1.05446696f, -1.03651285f, 0.41310427f, -1.10807168f, -0.06346543f, 1.12559628f, 0.22322938f, 0.74195981f, -1.39145792f, 0.43326408f, 0.77859634f, -1.12276006f};
const float B2[] = {-0.75145864f, -0.12630549f};

// Metadata
// input_cols: 254,550,tem
// target_cols: cod,uv254

#endif // MODEL_DATA_H
