#ifndef FASTREFLEX_PREPROCESSING_H
#define FASTREFLEX_PREPROCESSING_H

#include <stdbool.h>
#include <stdint.h>

#include "fastreflex_contract.h"

typedef struct
{
    float base_history[10][10];
    uint32_t base_count;
    uint32_t base_next;
    float normalized_ring[FASTREFLEX_HISTORY_SAMPLES][FASTREFLEX_FEATURE_DIMENSION];
    int8_t quantized_ring[FASTREFLEX_HISTORY_SAMPLES][FASTREFLEX_FEATURE_DIMENSION];
    uint32_t normalized_count;
    uint32_t normalized_next;
} fastreflex_preprocessor_t;

void fastreflex_preprocessor_reset(fastreflex_preprocessor_t *state);

void fastreflex_extract_causal(
    fastreflex_preprocessor_t *state,
    const float imu6[FASTREFLEX_RAW_DIMENSION],
    float causal[FASTREFLEX_FEATURE_DIMENSION]);

bool fastreflex_normalize_and_push(
    fastreflex_preprocessor_t *state,
    const float causal[FASTREFLEX_FEATURE_DIMENSION],
    float window[FASTREFLEX_WINDOW_ELEMENTS]);

bool fastreflex_push_normalized(
    fastreflex_preprocessor_t *state,
    const float normalized[FASTREFLEX_FEATURE_DIMENSION],
    float window[FASTREFLEX_WINDOW_ELEMENTS]);

void fastreflex_copy_quantized_window(
    const fastreflex_preprocessor_t *state,
    int8_t window[FASTREFLEX_WINDOW_ELEMENTS]);

#endif
