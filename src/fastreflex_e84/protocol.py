"""Versioned binary protocol shared by the E84 firmware and host HIL runner."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum


MAGIC = b"FR"
VERSION = 1
MAX_PAYLOAD_BYTES = 6400
HEADER = struct.Struct("<2sBBIIH")
CRC = struct.Struct("<I")
RESULT_PAYLOAD = struct.Struct("<4fBBH8I6I")


class PayloadType(IntEnum):
    WINDOW_FLOAT32 = 1
    FEATURE_FLOAT32 = 2
    RAW_IMU6_FLOAT32 = 3
    RESET_STATE = 16
    RESULT = 128


@dataclass(frozen=True)
class Packet:
    payload_type: PayloadType
    sequence_id: int
    window_endpoint: int
    payload: bytes = b""


@dataclass(frozen=True)
class RuntimeResult:
    member_probabilities: tuple[float, float, float]
    ensemble_probability: float
    threshold_crossing: bool
    persistence_count: int
    reflex_required: bool
    status_code: int
    receive_interval_us: int
    feature_us: int
    normalization_us: int
    member_inference_us: tuple[int, int, int]
    decision_us: int
    total_processing_us: int
    received_samples: int
    processed_samples: int
    dropped_samples: int
    queue_overruns: int
    deadline_misses: int
    sequence_gaps: int


def encode_packet(packet: Packet) -> bytes:
    if not 0 <= packet.sequence_id <= 0xFFFFFFFF:
        raise ValueError("sequence_id is outside uint32")
    if not 0 <= packet.window_endpoint <= 0xFFFFFFFF:
        raise ValueError("window endpoint is outside uint32")
    if len(packet.payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("payload exceeds the protocol maximum")
    header = HEADER.pack(
        MAGIC,
        VERSION,
        int(packet.payload_type),
        packet.sequence_id,
        packet.window_endpoint,
        len(packet.payload),
    )
    body = header + packet.payload
    return body + CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def decode_packet(frame: bytes) -> Packet:
    if len(frame) < HEADER.size + CRC.size:
        raise ValueError("truncated packet")
    magic, version, payload_type, sequence, endpoint, length = HEADER.unpack_from(frame)
    if magic != MAGIC or version != VERSION:
        raise ValueError("packet magic or version mismatch")
    expected_length = HEADER.size + length + CRC.size
    if len(frame) != expected_length or length > MAX_PAYLOAD_BYTES:
        raise ValueError("packet length mismatch")
    expected_crc = CRC.unpack_from(frame, HEADER.size + length)[0]
    if zlib.crc32(frame[: HEADER.size + length]) & 0xFFFFFFFF != expected_crc:
        raise ValueError("packet CRC mismatch")
    try:
        kind = PayloadType(payload_type)
    except ValueError as exc:
        raise ValueError(f"unknown payload type: {payload_type}") from exc
    return Packet(kind, sequence, endpoint, frame[HEADER.size : HEADER.size + length])


class FrameDecoder:
    """Incrementally recover CRC-protected frames from a serial byte stream."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.crc_errors = 0
        self.discarded_bytes = 0

    def feed(self, data: bytes) -> list[Packet]:
        self._buffer.extend(data)
        packets: list[Packet] = []
        while True:
            start = self._buffer.find(MAGIC)
            if start < 0:
                keep = 1 if self._buffer.endswith(MAGIC[:1]) else 0
                self.discarded_bytes += len(self._buffer) - keep
                if keep:
                    self._buffer[:] = self._buffer[-1:]
                else:
                    self._buffer.clear()
                break
            if start:
                self.discarded_bytes += start
                del self._buffer[:start]
            if len(self._buffer) < HEADER.size:
                break
            length = HEADER.unpack_from(self._buffer)[5]
            if length > MAX_PAYLOAD_BYTES:
                self.discarded_bytes += 1
                del self._buffer[0]
                continue
            frame_length = HEADER.size + length + CRC.size
            if len(self._buffer) < frame_length:
                break
            candidate = bytes(self._buffer[:frame_length])
            try:
                packets.append(decode_packet(candidate))
                del self._buffer[:frame_length]
            except ValueError:
                self.crc_errors += 1
                self.discarded_bytes += 1
                del self._buffer[0]
        return packets


class SequenceTracker:
    def __init__(self) -> None:
        self.expected: int | None = None
        self.gaps = 0
        self.out_of_order = 0

    def observe(self, sequence_id: int) -> None:
        if self.expected is None:
            self.expected = (sequence_id + 1) & 0xFFFFFFFF
            return
        delta = (sequence_id - self.expected) & 0xFFFFFFFF
        if delta == 0:
            self.expected = (sequence_id + 1) & 0xFFFFFFFF
        elif delta < 0x80000000:
            self.gaps += delta
            self.expected = (sequence_id + 1) & 0xFFFFFFFF
        else:
            self.out_of_order += 1


def encode_runtime_result(result: RuntimeResult) -> bytes:
    flags = int(result.threshold_crossing) | (int(result.reflex_required) << 1)
    return RESULT_PAYLOAD.pack(
        *result.member_probabilities,
        result.ensemble_probability,
        flags,
        result.persistence_count,
        result.status_code,
        result.receive_interval_us,
        result.feature_us,
        result.normalization_us,
        *result.member_inference_us,
        result.decision_us,
        result.total_processing_us,
        result.received_samples,
        result.processed_samples,
        result.dropped_samples,
        result.queue_overruns,
        result.deadline_misses,
        result.sequence_gaps,
    )


def decode_runtime_result(payload: bytes) -> RuntimeResult:
    if len(payload) != RESULT_PAYLOAD.size:
        raise ValueError("runtime result payload length mismatch")
    values = RESULT_PAYLOAD.unpack(payload)
    flags = values[4]
    return RuntimeResult(
        member_probabilities=(values[0], values[1], values[2]),
        ensemble_probability=values[3],
        threshold_crossing=bool(flags & 1),
        persistence_count=values[5],
        reflex_required=bool(flags & 2),
        status_code=values[6],
        receive_interval_us=values[7],
        feature_us=values[8],
        normalization_us=values[9],
        member_inference_us=(values[10], values[11], values[12]),
        decision_us=values[13],
        total_processing_us=values[14],
        received_samples=values[15],
        processed_samples=values[16],
        dropped_samples=values[17],
        queue_overruns=values[18],
        deadline_misses=values[19],
        sequence_gaps=values[20],
    )
