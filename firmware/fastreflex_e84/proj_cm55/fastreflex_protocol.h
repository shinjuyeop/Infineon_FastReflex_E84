#ifndef FASTREFLEX_PROTOCOL_H
#define FASTREFLEX_PROTOCOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mtb_hal.h"
#include "fastreflex_contract.h"

typedef enum
{
    FASTREFLEX_PAYLOAD_WINDOW_FLOAT32 = 1,
    FASTREFLEX_PAYLOAD_FEATURE_FLOAT32 = 2,
    FASTREFLEX_PAYLOAD_RAW_IMU6_FLOAT32 = 3,
    FASTREFLEX_PAYLOAD_RESET_STATE = 16,
    FASTREFLEX_PAYLOAD_RESULT = 128,
} fastreflex_payload_type_t;

typedef enum
{
    FASTREFLEX_STATUS_OK = 0,
    FASTREFLEX_STATUS_RESET_COMPLETE = 1,
    FASTREFLEX_STATUS_WINDOW_NOT_READY = 2,
    FASTREFLEX_STATUS_BAD_PAYLOAD = 10,
    FASTREFLEX_STATUS_BAD_PAYLOAD_TYPE = 11,
    FASTREFLEX_STATUS_MODEL_INFERENCE_ERROR = 20,
} fastreflex_status_t;

enum
{
    FASTREFLEX_RESULT_THRESHOLD_CROSSING = 1u << 0,
    FASTREFLEX_RESULT_REFLEX_REQUIRED = 1u << 1,
};

typedef struct
{
    uint8_t payload_type;
    uint32_t sequence_id;
    uint32_t window_endpoint;
    uint16_t payload_length;
} fastreflex_packet_t;

#pragma pack(push, 1)
typedef struct
{
    float member_probability[FASTREFLEX_MEMBER_COUNT];
    float ensemble_probability;
    uint8_t flags;
    uint8_t persistence_count;
    uint16_t status_code;
    uint32_t receive_interval_us;
    uint32_t feature_us;
    uint32_t normalization_us;
    uint32_t member_inference_us[FASTREFLEX_MEMBER_COUNT];
    uint32_t decision_us;
    uint32_t total_processing_us;
} fastreflex_runtime_result_t;

typedef struct
{
    uint32_t received_samples;
    uint32_t processed_samples;
    uint32_t dropped_samples;
    uint32_t queue_overruns;
    uint32_t deadline_misses;
    uint32_t sequence_gaps;
    uint32_t crc_errors;
} fastreflex_counters_t;
#pragma pack(pop)

bool fastreflex_receive_packet(
    mtb_hal_uart_t *uart,
    fastreflex_packet_t *packet,
    uint8_t *payload,
    size_t capacity,
    uint32_t *crc_errors);

bool fastreflex_send_result(
    mtb_hal_uart_t *uart,
    const fastreflex_packet_t *request,
    const fastreflex_runtime_result_t *result,
    const fastreflex_counters_t *counters);

#endif
