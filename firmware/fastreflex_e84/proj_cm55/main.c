/*
 * FastReflex non-release E84 HIL runtime.
 *
 * This firmware validates a deployment path only. It does not change the
 * formal INT8 numerical-contract failure or authorize M4/release use.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "cybsp.h"
#include "mtb_ml.h"

#include "app_common.h"
#include "fastreflex_preprocessing.h"
#include "fastreflex_protocol.h"
#include "FASTREFLEX_SEED20260828_tflm_model_int8x8.h"
#include "FASTREFLEX_SEED20260829_tflm_model_int8x8.h"
#include "FASTREFLEX_SEED20260830_tflm_model_int8x8.h"

#define FASTREFLEX_UART_BAUD              (1000000u)
#define FASTREFLEX_ML_PRIORITY            (3u)
#define FASTREFLEX_OUTPUT_SCALE           (0.00390625f)
#define FASTREFLEX_OUTPUT_ZERO_POINT      (-128)
#define FASTREFLEX_THRESHOLD              (0.99f)
#define FASTREFLEX_PERSISTENCE_SAMPLES    (5u)
#define FASTREFLEX_DEADLINE_US            (1000u)

static uint8_t payload_buffer[FASTREFLEX_MAX_PAYLOAD_BYTES]
    __attribute__((aligned(4)));
static float model_window[FASTREFLEX_WINDOW_ELEMENTS]
    __attribute__((aligned(16)));
static int8_t quantized_window[FASTREFLEX_WINDOW_ELEMENTS]
    __attribute__((aligned(16)));

static mtb_ml_model_t *models[FASTREFLEX_MEMBER_COUNT];
static MTB_ML_DATA_T *model_outputs[FASTREFLEX_MEMBER_COUNT];
static fastreflex_preprocessor_t preprocessor;
static fastreflex_counters_t counters;
static uint32_t previous_sequence;
static bool have_previous_sequence;
static uint32_t previous_receive_cycle;
static bool have_previous_receive_cycle;
static uint8_t persistence_count;

static const mtb_ml_model_bin_t model_bins[FASTREFLEX_MEMBER_COUNT] = {
    {
        "FASTREFLEX_SEED20260828",
        FASTREFLEX_SEED20260828_model_bin,
        FASTREFLEX_SEED20260828_MODEL_BIN_LEN,
        FASTREFLEX_SEED20260828_ARENA_SIZE,
    },
    {
        "FASTREFLEX_SEED20260829",
        FASTREFLEX_SEED20260829_model_bin,
        FASTREFLEX_SEED20260829_MODEL_BIN_LEN,
        FASTREFLEX_SEED20260829_ARENA_SIZE,
    },
    {
        "FASTREFLEX_SEED20260830",
        FASTREFLEX_SEED20260830_model_bin,
        FASTREFLEX_SEED20260830_MODEL_BIN_LEN,
        FASTREFLEX_SEED20260830_ARENA_SIZE,
    },
};

static void timer_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0u;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static uint32_t timer_cycles(void)
{
    return DWT->CYCCNT;
}

static uint32_t cycles_to_us(uint32_t cycles)
{
    uint32_t cycles_per_us = SystemCoreClock / 1000000u;
    return (cycles + (cycles_per_us / 2u)) / cycles_per_us;
}

static void reset_runtime_state(void)
{
    fastreflex_preprocessor_reset(&preprocessor);
    memset(&counters, 0, sizeof(counters));
    persistence_count = 0u;
    have_previous_sequence = false;
    have_previous_receive_cycle = false;
}

static bool model_contract_is_valid(const mtb_ml_model_t *model)
{
    return model != NULL
        && model->input_size == (int)FASTREFLEX_WINDOW_ELEMENTS
        && model->output_size == 2
        && model->input_type_size == 1
        && model->output_type_size == 1
        && fabsf(model->input_scale - FASTREFLEX_INPUT_SCALE) < 1.0e-9f
        && model->input_zero_point == 0
        && fabsf(model->output_scale - FASTREFLEX_OUTPUT_SCALE) < 1.0e-9f
        && model->output_zero_point == FASTREFLEX_OUTPUT_ZERO_POINT;
}

static cy_rslt_t models_init(void)
{
    cy_rslt_t result = mtb_ml_init(FASTREFLEX_ML_PRIORITY);
    if (result != CY_RSLT_SUCCESS)
    {
        return result;
    }
    for (uint32_t index = 0u; index < FASTREFLEX_MEMBER_COUNT; ++index)
    {
        int output_size = 0;
        result = mtb_ml_model_init(&model_bins[index], NULL, &models[index]);
        if (result != CY_RSLT_SUCCESS || !model_contract_is_valid(models[index]))
        {
            return result == CY_RSLT_SUCCESS ? MTB_ML_RESULT_BAD_MODEL : result;
        }
        result = mtb_ml_model_get_output(
            models[index], &model_outputs[index], &output_size);
        if (result != CY_RSLT_SUCCESS || output_size != 2)
        {
            return result == CY_RSLT_SUCCESS ? MTB_ML_RESULT_BAD_MODEL : result;
        }
    }
    return CY_RSLT_SUCCESS;
}

static void quantize_window(void)
{
    for (uint32_t index = 0u; index < FASTREFLEX_WINDOW_ELEMENTS; ++index)
    {
        long value = lrintf(model_window[index] / FASTREFLEX_INPUT_SCALE);
        if (value < -128L)
        {
            value = -128L;
        }
        else if (value > 127L)
        {
            value = 127L;
        }
        quantized_window[index] = (int8_t)value;
    }
}

static uint16_t run_ensemble(
    fastreflex_runtime_result_t *response, bool input_is_quantized)
{
    if (!input_is_quantized)
    {
        quantize_window();
    }
    float sum = 0.0f;
    for (uint32_t index = 0u; index < FASTREFLEX_MEMBER_COUNT; ++index)
    {
        uint32_t start = timer_cycles();
        cy_rslt_t result = mtb_ml_model_run(models[index], quantized_window);
        response->member_inference_us[index] = cycles_to_us(timer_cycles() - start);
        if (result != CY_RSLT_SUCCESS)
        {
            return FASTREFLEX_STATUS_MODEL_INFERENCE_ERROR;
        }
        int32_t hazard_quantized = ((int8_t *)model_outputs[index])[1];
        response->member_probability[index] =
            (float)(hazard_quantized - FASTREFLEX_OUTPUT_ZERO_POINT)
            * FASTREFLEX_OUTPUT_SCALE;
        sum += response->member_probability[index];
    }
    response->ensemble_probability = sum / (float)FASTREFLEX_MEMBER_COUNT;
    return FASTREFLEX_STATUS_OK;
}

static void update_sequence(uint32_t sequence)
{
    if (have_previous_sequence)
    {
        uint32_t expected = previous_sequence + 1u;
        uint32_t delta = sequence - expected;
        if (delta != 0u && delta < UINT32_C(0x80000000))
        {
            counters.sequence_gaps += delta;
            counters.dropped_samples += delta;
        }
    }
    previous_sequence = sequence;
    have_previous_sequence = true;
}

static uint16_t prepare_window(
    const fastreflex_packet_t *packet,
    fastreflex_runtime_result_t *response,
    bool *input_is_quantized)
{
    uint32_t start;
    *input_is_quantized = false;
    if (packet->payload_type == FASTREFLEX_PAYLOAD_WINDOW_FLOAT32)
    {
        if (packet->payload_length != sizeof(model_window))
        {
            return FASTREFLEX_STATUS_BAD_PAYLOAD;
        }
        memcpy(model_window, payload_buffer, sizeof(model_window));
        return FASTREFLEX_STATUS_OK;
    }
    if (packet->payload_type == FASTREFLEX_PAYLOAD_FEATURE_FLOAT32)
    {
        if (packet->payload_length != FASTREFLEX_FEATURE_DIMENSION * sizeof(float))
        {
            return FASTREFLEX_STATUS_BAD_PAYLOAD;
        }
        start = timer_cycles();
        bool ready = fastreflex_push_normalized(
            &preprocessor, (const float *)payload_buffer, NULL);
        if (ready)
        {
            fastreflex_copy_quantized_window(&preprocessor, quantized_window);
            *input_is_quantized = true;
        }
        response->normalization_us = cycles_to_us(timer_cycles() - start);
        return ready ? FASTREFLEX_STATUS_OK : FASTREFLEX_STATUS_WINDOW_NOT_READY;
    }
    if (packet->payload_type == FASTREFLEX_PAYLOAD_RAW_IMU6_FLOAT32)
    {
        float causal[FASTREFLEX_FEATURE_DIMENSION];
        if (packet->payload_length != FASTREFLEX_RAW_DIMENSION * sizeof(float))
        {
            return FASTREFLEX_STATUS_BAD_PAYLOAD;
        }
        start = timer_cycles();
        fastreflex_extract_causal(
            &preprocessor, (const float *)payload_buffer, causal);
        response->feature_us = cycles_to_us(timer_cycles() - start);
        start = timer_cycles();
        bool ready = fastreflex_normalize_and_push(
            &preprocessor, causal, NULL);
        if (ready)
        {
            fastreflex_copy_quantized_window(&preprocessor, quantized_window);
            *input_is_quantized = true;
        }
        response->normalization_us = cycles_to_us(timer_cycles() - start);
        return ready ? FASTREFLEX_STATUS_OK : FASTREFLEX_STATUS_WINDOW_NOT_READY;
    }
    return FASTREFLEX_STATUS_BAD_PAYLOAD_TYPE;
}

static void process_packet(const fastreflex_packet_t *packet)
{
    fastreflex_runtime_result_t response;
    bool input_is_quantized = false;
    uint32_t total_start = timer_cycles();
    memset(&response, 0, sizeof(response));

    if (packet->payload_type == FASTREFLEX_PAYLOAD_RESET_STATE)
    {
        reset_runtime_state();
        response.status_code = FASTREFLEX_STATUS_RESET_COMPLETE;
        fastreflex_send_result(
            &mtb_ml_retarget_io_uart_obj, packet, &response, &counters);
        return;
    }

    counters.received_samples++;
    update_sequence(packet->sequence_id);
    uint32_t receive_cycle = timer_cycles();
    if (have_previous_receive_cycle)
    {
        response.receive_interval_us =
            cycles_to_us(receive_cycle - previous_receive_cycle);
    }
    previous_receive_cycle = receive_cycle;
    have_previous_receive_cycle = true;

    response.status_code = prepare_window(packet, &response, &input_is_quantized);
    counters.processed_samples++;
    if (response.status_code == FASTREFLEX_STATUS_OK)
    {
        response.status_code = run_ensemble(&response, input_is_quantized);
    }
    if (response.status_code == FASTREFLEX_STATUS_OK)
    {
        uint32_t decision_start = timer_cycles();
        bool crossing = response.ensemble_probability >= FASTREFLEX_THRESHOLD;
        persistence_count = crossing ? (uint8_t)(persistence_count + 1u) : 0u;
        response.flags = crossing ? FASTREFLEX_RESULT_THRESHOLD_CROSSING : 0u;
        if (persistence_count >= FASTREFLEX_PERSISTENCE_SAMPLES)
        {
            response.flags |= FASTREFLEX_RESULT_REFLEX_REQUIRED;
        }
        response.persistence_count = persistence_count;
        response.decision_us = cycles_to_us(timer_cycles() - decision_start);
    }
    response.total_processing_us = cycles_to_us(timer_cycles() - total_start);
    if (response.total_processing_us > FASTREFLEX_DEADLINE_US)
    {
        counters.deadline_misses++;
    }
    fastreflex_send_result(
        &mtb_ml_retarget_io_uart_obj, packet, &response, &counters);
}

int main(void)
{
    if (cybsp_init() != CY_RSLT_SUCCESS)
    {
        CY_ASSERT(0);
    }
    __enable_irq();
    app_retarget_io_init(FASTREFLEX_UART_BAUD);
    timer_init();
    reset_runtime_state();
    if (models_init() != CY_RSLT_SUCCESS)
    {
        handle_error();
    }

    for (;;)
    {
        fastreflex_packet_t packet;
        if (fastreflex_receive_packet(
                &mtb_ml_retarget_io_uart_obj,
                &packet,
                payload_buffer,
                sizeof(payload_buffer),
                &counters.crc_errors))
        {
            process_packet(&packet);
        }
    }
}
