#include "fastreflex_protocol.h"

#include <string.h>

#define FASTREFLEX_HEADER_BYTES    (14u)
#define FASTREFLEX_CRC_BYTES       (4u)
#define FASTREFLEX_RESULT_BYTES    (76u)
#define FASTREFLEX_INTERBYTE_TIMEOUT_MS (10u)

_Static_assert(sizeof(fastreflex_runtime_result_t) == 52u, "result layout changed");

static uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t length)
{
    for (size_t index = 0u; index < length; ++index)
    {
        crc ^= data[index];
        for (uint32_t bit = 0u; bit < 8u; ++bit)
        {
            uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1u) ^ (UINT32_C(0xedb88320) & mask);
        }
    }
    return crc;
}

static uint16_t get_u16(const uint8_t *value)
{
    return (uint16_t)value[0] | ((uint16_t)value[1] << 8u);
}

static uint32_t get_u32(const uint8_t *value)
{
    return (uint32_t)value[0]
        | ((uint32_t)value[1] << 8u)
        | ((uint32_t)value[2] << 16u)
        | ((uint32_t)value[3] << 24u);
}

static void put_u16(uint8_t *value, uint16_t input)
{
    value[0] = (uint8_t)input;
    value[1] = (uint8_t)(input >> 8u);
}

static void put_u32(uint8_t *value, uint32_t input)
{
    value[0] = (uint8_t)input;
    value[1] = (uint8_t)(input >> 8u);
    value[2] = (uint8_t)(input >> 16u);
    value[3] = (uint8_t)(input >> 24u);
}

static bool uart_read_exact(mtb_hal_uart_t *uart, uint8_t *value, size_t length)
{
    for (size_t index = 0u; index < length; ++index)
    {
        if (mtb_hal_uart_get(
                uart, &value[index], FASTREFLEX_INTERBYTE_TIMEOUT_MS)
            != CY_RSLT_SUCCESS)
        {
            return false;
        }
    }
    return true;
}

static bool uart_write_all(mtb_hal_uart_t *uart, const void *value, size_t length)
{
    const uint8_t *cursor = (const uint8_t *)value;
    while (length > 0u)
    {
        size_t chunk = length;
        if (mtb_hal_uart_write(uart, (void *)cursor, &chunk) != CY_RSLT_SUCCESS)
        {
            return false;
        }
        if (chunk == 0u)
        {
            return false;
        }
        cursor += chunk;
        length -= chunk;
    }
    return true;
}

bool fastreflex_receive_packet(
    mtb_hal_uart_t *uart,
    fastreflex_packet_t *packet,
    uint8_t *payload,
    size_t capacity,
    uint32_t *crc_errors)
{
    uint8_t header[FASTREFLEX_HEADER_BYTES];
    uint8_t byte = 0u;
    bool have_f = false;
    for (;;)
    {
        if (!uart_read_exact(uart, &byte, 1u))
        {
            return false;
        }
        if (have_f && byte == (uint8_t)'R')
        {
            header[0] = (uint8_t)'F';
            header[1] = (uint8_t)'R';
            break;
        }
        have_f = byte == (uint8_t)'F';
    }
    if (!uart_read_exact(uart, &header[2], FASTREFLEX_HEADER_BYTES - 2u))
    {
        return false;
    }
    uint16_t payload_length = get_u16(&header[12]);
    if (header[2] != FASTREFLEX_PROTOCOL_VERSION
        || payload_length > capacity
        || payload_length > FASTREFLEX_MAX_PAYLOAD_BYTES)
    {
        return false;
    }
    if (!uart_read_exact(uart, payload, payload_length))
    {
        return false;
    }
    uint8_t crc_bytes[FASTREFLEX_CRC_BYTES];
    if (!uart_read_exact(uart, crc_bytes, sizeof(crc_bytes)))
    {
        return false;
    }
    uint32_t crc = crc32_update(UINT32_C(0xffffffff), header, sizeof(header));
    crc = crc32_update(crc, payload, payload_length) ^ UINT32_C(0xffffffff);
    if (crc != get_u32(crc_bytes))
    {
        if (crc_errors != NULL)
        {
            (*crc_errors)++;
        }
        return false;
    }
    packet->payload_type = header[3];
    packet->sequence_id = get_u32(&header[4]);
    packet->window_endpoint = get_u32(&header[8]);
    packet->payload_length = payload_length;
    return true;
}

bool fastreflex_send_result(
    mtb_hal_uart_t *uart,
    const fastreflex_packet_t *request,
    const fastreflex_runtime_result_t *result,
    const fastreflex_counters_t *counters)
{
    uint8_t header[FASTREFLEX_HEADER_BYTES] = {
        (uint8_t)'F', (uint8_t)'R', FASTREFLEX_PROTOCOL_VERSION,
        FASTREFLEX_PAYLOAD_RESULT,
    };
    uint8_t payload[FASTREFLEX_RESULT_BYTES];
    memcpy(payload, result, sizeof(*result));
    memcpy(payload + sizeof(*result), counters, 6u * sizeof(uint32_t));
    put_u32(&header[4], request->sequence_id);
    put_u32(&header[8], request->window_endpoint);
    put_u16(&header[12], sizeof(payload));
    uint32_t crc = crc32_update(UINT32_C(0xffffffff), header, sizeof(header));
    crc = crc32_update(crc, payload, sizeof(payload)) ^ UINT32_C(0xffffffff);
    uint8_t crc_bytes[FASTREFLEX_CRC_BYTES];
    put_u32(crc_bytes, crc);
    return uart_write_all(uart, header, sizeof(header))
        && uart_write_all(uart, payload, sizeof(payload))
        && uart_write_all(uart, crc_bytes, sizeof(crc_bytes));
}
