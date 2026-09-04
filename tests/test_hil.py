from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fastreflex_e84.hil import DecisionState, HILEvaluator
from fastreflex_e84.preprocessing import StreamingPreprocessor
from fastreflex_e84.protocol import (
    FrameDecoder,
    Packet,
    PayloadType,
    RuntimeResult,
    SequenceTracker,
    decode_packet,
    decode_runtime_result,
    encode_packet,
    encode_runtime_result,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "model/source/model_v2_anchor_refined_gru20_20260902"


def runtime_result(**overrides: object) -> RuntimeResult:
    values: dict[str, object] = {
        "member_probabilities": (0.25, 0.5, 0.75),
        "ensemble_probability": 0.5,
        "threshold_crossing": False,
        "persistence_count": 0,
        "reflex_required": False,
        "status_code": 0,
        "receive_interval_us": 1000,
        "feature_us": 8,
        "normalization_us": 3,
        "member_inference_us": (100, 101, 99),
        "decision_us": 2,
        "total_processing_us": 313,
        "received_samples": 10,
        "processed_samples": 10,
        "dropped_samples": 0,
        "queue_overruns": 0,
        "deadline_misses": 0,
        "sequence_gaps": 0,
    }
    values.update(overrides)
    return RuntimeResult(**values)


def test_protocol_round_trip_crc_and_incremental_decode() -> None:
    result = runtime_result(threshold_crossing=True, reflex_required=True)
    packet = Packet(PayloadType.RESULT, 42, 107, encode_runtime_result(result))
    frame = encode_packet(packet)
    decoded = decode_packet(frame)
    assert decoded == packet
    actual = decode_runtime_result(decoded.payload)
    assert actual.member_probabilities == pytest.approx((0.25, 0.5, 0.75))
    assert actual.threshold_crossing is True
    assert actual.reflex_required is True
    assert actual.member_inference_us == (100, 101, 99)

    decoder = FrameDecoder()
    assert decoder.feed(b"noise" + frame[:7]) == []
    assert decoder.feed(frame[7:31]) == []
    assert decoder.feed(frame[31:]) == [packet]
    assert decoder.discarded_bytes == 5

    corrupt = bytearray(frame)
    corrupt[-1] ^= 0x80
    with pytest.raises(ValueError, match="CRC"):
        decode_packet(bytes(corrupt))


def test_decision_and_sequence_state_are_fail_closed() -> None:
    decision = DecisionState()
    values = [0.99, 1.0, 0.991, 0.999, 0.99, 0.2]
    outputs = [decision.update(value) for value in values]
    assert [row[1] for row in outputs] == [1, 2, 3, 4, 5, 0]
    assert [row[2] for row in outputs] == [False, False, False, False, True, False]
    assert [row[3] for row in outputs] == [False, False, False, False, True, False]

    sequence = SequenceTracker()
    for value in (100, 101, 104, 103, 105):
        sequence.observe(value)
    assert sequence.gaps == 2
    assert sequence.out_of_order == 1


def test_streaming_preprocessing_matches_all_golden_layers_and_ring_buffer() -> None:
    with np.load(
        BUNDLE / "golden_inputs/runtime_chain.npz", allow_pickle=False
    ) as data:
        raw = data["raw_pelvis_imu6"].copy()
    with np.load(
        BUNDLE / "golden_outputs/deployment_runtime_chain.npz", allow_pickle=False
    ) as data:
        golden = {name: data[name].copy() for name in data.files}

    preprocessor = StreamingPreprocessor.from_bundle(BUNDLE)
    steps = [preprocessor.push(sample) for sample in raw]
    base = np.stack([step.base for step in steps])
    causal = np.stack([step.causal for step in steps])
    normalized = np.stack([step.normalized for step in steps])
    windows = np.stack([step.window for step in steps if step.window is not None])

    assert all(step.window is None for step in steps[:19])
    assert steps[19].window is not None
    assert windows.shape == (121, 20, 80)
    assert np.max(np.abs(base - golden["base_features"])) <= 1.0e-6
    assert np.max(np.abs(causal - golden["causal_features"])) <= 1.0e-6
    assert np.max(np.abs(normalized - golden["normalized_features"])) <= 1.5e-6
    assert np.max(np.abs(windows - golden["model_windows"])) <= 1.5e-6
    assert np.array_equal(windows[-1, -1], normalized[-1])


def test_hil_evaluator_detects_sequence_loss_and_board_drop() -> None:
    evaluator = HILEvaluator()
    first = Packet(PayloadType.RESULT, 10, 19, encode_runtime_result(runtime_result()))
    second = Packet(
        PayloadType.RESULT,
        12,
        21,
        encode_runtime_result(runtime_result(dropped_samples=1, sequence_gaps=1)),
    )
    evaluator.observe(first, 1_000_000, 1_300_000)
    evaluator.observe(second, 2_000_000, 2_400_000)
    report = evaluator.summary(sent_count=3, sender_deadline_misses=0)
    assert report["status"] == "FAIL"
    assert report["missing_results"] == 1
    assert report["host_sequence_gaps"] == 1
    assert report["board_counters"]["dropped_samples"] == 1
