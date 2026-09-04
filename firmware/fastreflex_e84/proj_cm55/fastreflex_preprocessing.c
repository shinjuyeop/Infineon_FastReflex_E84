#include "fastreflex_preprocessing.h"

#include <math.h>
#include <string.h>

#include "fastreflex_normalizer.h"

#define BASE_DIMENSION       (10u)
#define BASE_HISTORY         (10u)

static int8_t quantize_normalized(float value)
{
    long quantized = lrintf(value / FASTREFLEX_INPUT_SCALE);
    if (quantized < -128L)
    {
        quantized = -128L;
    }
    else if (quantized > 127L)
    {
        quantized = 127L;
    }
    return (int8_t)quantized;
}

void fastreflex_preprocessor_reset(fastreflex_preprocessor_t *state)
{
    memset(state, 0, sizeof(*state));
}

static const float *previous_base(
    const fastreflex_preprocessor_t *state, uint32_t lag)
{
    uint32_t index = (state->base_next + BASE_HISTORY - lag) % BASE_HISTORY;
    return state->base_history[index];
}

static void rolling_statistics(
    const fastreflex_preprocessor_t *state,
    const float current[BASE_DIMENSION],
    uint32_t width,
    float mean[BASE_DIMENSION],
    float variance[BASE_DIMENSION])
{
    uint32_t prior_count = state->base_count;
    if (prior_count > width - 1u)
    {
        prior_count = width - 1u;
    }
    double count = (double)(prior_count + 1u);
    for (uint32_t feature = 0u; feature < BASE_DIMENSION; ++feature)
    {
        double sum = current[feature];
        double sum_square = (double)current[feature] * current[feature];
        for (uint32_t lag = 1u; lag <= prior_count; ++lag)
        {
            double value = previous_base(state, lag)[feature];
            sum += value;
            sum_square += value * value;
        }
        double average = sum / count;
        double observed_variance = (sum_square / count) - (average * average);
        mean[feature] = (float)average;
        variance[feature] = (float)(observed_variance > 0.0 ? observed_variance : 0.0);
    }
}

void fastreflex_extract_causal(
    fastreflex_preprocessor_t *state,
    const float imu6[FASTREFLEX_RAW_DIMENSION],
    float causal[FASTREFLEX_FEATURE_DIMENSION])
{
    float base[BASE_DIMENSION];
    memcpy(base, imu6, FASTREFLEX_RAW_DIMENSION * sizeof(float));
    base[6] = sqrtf(imu6[0] * imu6[0] + imu6[1] * imu6[1] + imu6[2] * imu6[2]);
    base[7] = sqrtf(imu6[3] * imu6[3] + imu6[4] * imu6[4] + imu6[5] * imu6[5]);
    base[8] = sqrtf(imu6[0] * imu6[0] + imu6[1] * imu6[1]);
    base[9] = sqrtf(imu6[3] * imu6[3] + imu6[4] * imu6[4]);
    memcpy(&causal[0], base, sizeof(base));

    const uint32_t lags[3] = {1u, 5u, 10u};
    for (uint32_t group = 0u; group < 3u; ++group)
    {
        float *delta = &causal[(group + 1u) * BASE_DIMENSION];
        if (state->base_count < lags[group])
        {
            memset(delta, 0, BASE_DIMENSION * sizeof(float));
        }
        else
        {
            const float *previous = previous_base(state, lags[group]);
            for (uint32_t feature = 0u; feature < BASE_DIMENSION; ++feature)
            {
                delta[feature] = base[feature] - previous[feature];
            }
        }
    }
    rolling_statistics(state, base, 5u, &causal[40], &causal[60]);
    rolling_statistics(state, base, 10u, &causal[50], &causal[70]);

    memcpy(state->base_history[state->base_next], base, sizeof(base));
    state->base_next = (state->base_next + 1u) % BASE_HISTORY;
    if (state->base_count < BASE_HISTORY)
    {
        state->base_count++;
    }
}

bool fastreflex_push_normalized(
    fastreflex_preprocessor_t *state,
    const float normalized[FASTREFLEX_FEATURE_DIMENSION],
    float window[FASTREFLEX_WINDOW_ELEMENTS])
{
    memcpy(
        state->normalized_ring[state->normalized_next],
        normalized,
        FASTREFLEX_FEATURE_DIMENSION * sizeof(float));
    for (uint32_t feature = 0u; feature < FASTREFLEX_FEATURE_DIMENSION; ++feature)
    {
        state->quantized_ring[state->normalized_next][feature] =
            quantize_normalized(normalized[feature]);
    }
    state->normalized_next =
        (state->normalized_next + 1u) % FASTREFLEX_HISTORY_SAMPLES;
    if (state->normalized_count < FASTREFLEX_HISTORY_SAMPLES)
    {
        state->normalized_count++;
    }
    if (state->normalized_count < FASTREFLEX_HISTORY_SAMPLES)
    {
        return false;
    }
    if (window != NULL)
    {
        for (uint32_t timestep = 0u; timestep < FASTREFLEX_HISTORY_SAMPLES; ++timestep)
        {
            uint32_t source =
                (state->normalized_next + timestep) % FASTREFLEX_HISTORY_SAMPLES;
            memcpy(
                &window[timestep * FASTREFLEX_FEATURE_DIMENSION],
                state->normalized_ring[source],
                FASTREFLEX_FEATURE_DIMENSION * sizeof(float));
        }
    }
    return true;
}

void fastreflex_copy_quantized_window(
    const fastreflex_preprocessor_t *state,
    int8_t window[FASTREFLEX_WINDOW_ELEMENTS])
{
    for (uint32_t timestep = 0u; timestep < FASTREFLEX_HISTORY_SAMPLES; ++timestep)
    {
        uint32_t source =
            (state->normalized_next + timestep) % FASTREFLEX_HISTORY_SAMPLES;
        memcpy(
            &window[timestep * FASTREFLEX_FEATURE_DIMENSION],
            state->quantized_ring[source],
            FASTREFLEX_FEATURE_DIMENSION * sizeof(int8_t));
    }
}

bool fastreflex_normalize_and_push(
    fastreflex_preprocessor_t *state,
    const float causal[FASTREFLEX_FEATURE_DIMENSION],
    float window[FASTREFLEX_WINDOW_ELEMENTS])
{
    float normalized[FASTREFLEX_FEATURE_DIMENSION];
    for (uint32_t feature = 0u; feature < FASTREFLEX_FEATURE_DIMENSION; ++feature)
    {
        normalized[feature] =
            (causal[feature] - fastreflex_normalizer_mean[feature])
            / fastreflex_normalizer_std[feature];
    }
    return fastreflex_push_normalized(state, normalized, window);
}
