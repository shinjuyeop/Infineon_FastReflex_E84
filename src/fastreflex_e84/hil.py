"""Host-side replay, timing collection, and fail-closed HIL evaluation."""

from __future__ import annotations

import hashlib
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .protocol import (
    FrameDecoder,
    Packet,
    PayloadType,
    RuntimeResult,
    SequenceTracker,
    decode_runtime_result,
    encode_packet,
)


@dataclass(frozen=True)
class ObservedResult:
    packet: Packet
    result: RuntimeResult
    host_send_ns: int
    host_receive_ns: int


class DecisionState:
    def __init__(self, threshold: float = 0.99, persistence_samples: int = 5) -> None:
        if persistence_samples <= 0:
            raise ValueError("persistence must be positive")
        self.threshold = threshold
        self.persistence_samples = persistence_samples
        self.count = 0
        self.reflex_required = False

    def update(self, probability: float) -> tuple[bool, int, bool, bool]:
        crossing = probability >= self.threshold
        self.count = self.count + 1 if crossing else 0
        previous = self.reflex_required
        self.reflex_required = self.count >= self.persistence_samples
        return (
            crossing,
            self.count,
            self.reflex_required,
            self.reflex_required and not previous,
        )


def percentile(values: list[float], level: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), level))


def timing_statistics(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "standard_deviation": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "p95": percentile(values, 95),
        "maximum": max(values),
    }


class HILEvaluator:
    def __init__(self, expected: dict[str, np.ndarray] | None = None) -> None:
        self.expected = expected
        self.sequence = SequenceTracker()
        self.records: list[ObservedResult] = []

    def observe(self, packet: Packet, host_send_ns: int, host_receive_ns: int) -> None:
        if packet.payload_type != PayloadType.RESULT:
            raise ValueError("HIL evaluator accepts RESULT packets only")
        result = decode_runtime_result(packet.payload)
        if result.status_code == 1:
            self.sequence = SequenceTracker()
        else:
            self.sequence.observe(packet.sequence_id)
        self.records.append(
            ObservedResult(
                packet,
                result,
                host_send_ns,
                host_receive_ns,
            )
        )

    def summary(self, sent_count: int, sender_deadline_misses: int) -> dict[str, Any]:
        data_records = [row for row in self.records if row.result.status_code != 1]
        trace_digest = hashlib.sha256()
        for row in data_records:
            flags = int(row.result.threshold_crossing) | (
                int(row.result.reflex_required) << 1
            )
            trace_digest.update(
                struct.pack(
                    "<I4fBBH",
                    row.packet.window_endpoint,
                    *row.result.member_probabilities,
                    row.result.ensemble_probability,
                    flags,
                    row.result.persistence_count,
                    row.result.status_code,
                )
            )
        host_latency = [
            (row.host_receive_ns - row.host_send_ns) / 1000.0 for row in data_records
        ]
        send_times = [row.host_send_ns for row in data_records]
        send_intervals = [
            (right - left) / 1000.0 for left, right in zip(send_times, send_times[1:])
        ]
        latest = self.records[-1].result if self.records else None
        status_codes: dict[int, int] = {}
        for row in self.records:
            status_codes[row.result.status_code] = (
                status_codes.get(row.result.status_code, 0) + 1
            )
        unexpected_statuses = sum(
            count for status, count in status_codes.items() if status not in (0, 1, 2)
        )
        report: dict[str, Any] = {
            "sent_packets": sent_count,
            "received_results": len(self.records),
            "missing_results": sent_count - len(self.records),
            "host_sequence_gaps": self.sequence.gaps,
            "host_out_of_order": self.sequence.out_of_order,
            "host_sender_deadline_misses": sender_deadline_misses,
            "status_codes": {
                str(key): value for key, value in sorted(status_codes.items())
            },
            "unexpected_status_results": unexpected_statuses,
            "deployment_trace_sha256": trace_digest.hexdigest(),
            "host_send_interval_us": timing_statistics(send_intervals),
            "response_latency_us": timing_statistics(host_latency),
            "board_counters": None
            if latest is None
            else {
                "received_samples": latest.received_samples,
                "processed_samples": latest.processed_samples,
                "dropped_samples": latest.dropped_samples,
                "queue_overruns": latest.queue_overruns,
                "deadline_misses": latest.deadline_misses,
                "sequence_gaps": latest.sequence_gaps,
            },
            "board_timing_us": self._timing_summary(),
            "parity": self._parity(),
        }
        counters = report["board_counters"]
        report["status"] = (
            "PASS"
            if self.records
            and report["missing_results"] == 0
            and self.sequence.gaps == 0
            and self.sequence.out_of_order == 0
            and sender_deadline_misses == 0
            and unexpected_statuses == 0
            and counters is not None
            and counters["dropped_samples"] == 0
            and counters["queue_overruns"] == 0
            and counters["deadline_misses"] == 0
            and report["parity"]["status"] == "PASS"
            else "FAIL"
        )
        return report

    def _timing_summary(self) -> dict[str, Any]:
        data_records = [row for row in self.records if row.result.status_code != 1]
        inference_records = [row for row in data_records if row.result.status_code == 0]
        fields = {
            "receive_interval": [
                row.result.receive_interval_us for row in data_records
            ],
            "feature": [row.result.feature_us for row in data_records],
            "normalization": [row.result.normalization_us for row in data_records],
            "member_1": [
                row.result.member_inference_us[0] for row in inference_records
            ],
            "member_2": [
                row.result.member_inference_us[1] for row in inference_records
            ],
            "member_3": [
                row.result.member_inference_us[2] for row in inference_records
            ],
            "three_member_inference": [
                sum(row.result.member_inference_us) for row in inference_records
            ],
            "decision": [row.result.decision_us for row in inference_records],
            "total_processing": [
                row.result.total_processing_us for row in data_records
            ],
            "inference_total_processing": [
                row.result.total_processing_us for row in inference_records
            ],
        }
        return {name: timing_statistics(values) for name, values in fields.items()}

    def _parity(self) -> dict[str, Any]:
        if self.expected is None:
            return {"status": "PASS", "comparison": "not_requested"}
        endpoints = {
            int(endpoint): index
            for index, endpoint in enumerate(self.expected["window_endpoints"])
        }
        compared: list[tuple[ObservedResult, int]] = [
            (row, endpoints[row.packet.window_endpoint])
            for row in self.records
            if row.packet.window_endpoint in endpoints and row.result.status_code == 0
        ]
        if not compared:
            return {"status": "FAIL", "comparison": "no_comparable_results"}
        member_errors = []
        ensemble_errors = []
        crossing_mismatches = 0
        count_mismatches = 0
        reflex_mismatches = 0
        onset_mismatches = 0
        previous_reflex = False
        for row, index in compared:
            member_errors.extend(
                np.abs(
                    np.asarray(row.result.member_probabilities, dtype=np.float64)
                    - self.expected["member_hazard_probability"][:, index]
                ).tolist()
            )
            ensemble_errors.append(
                abs(
                    row.result.ensemble_probability
                    - float(self.expected["ensemble_hazard_probability"][index])
                )
            )
            crossing_mismatches += row.result.threshold_crossing != bool(
                self.expected["threshold_crossing"][index]
            )
            count_mismatches += row.result.persistence_count != int(
                self.expected["consecutive_threshold_count"][index]
            )
            reflex_mismatches += row.result.reflex_required != bool(
                self.expected["reflex_required"][index]
            )
            observed_onset = row.result.reflex_required and not previous_reflex
            onset_mismatches += observed_onset != bool(
                self.expected["reflex_onset"][index]
            )
            previous_reflex = row.result.reflex_required
        member_maximum = max(member_errors)
        ensemble_maximum = max(ensemble_errors)
        numerical_tolerance = 0.00390625
        numerical_status = (
            "PASS"
            if member_maximum <= numerical_tolerance
            and ensemble_maximum <= numerical_tolerance
            else "FAIL"
        )
        mismatch_count = (
            crossing_mismatches
            + count_mismatches
            + reflex_mismatches
            + onset_mismatches
        )
        return {
            "status": (
                "PASS" if mismatch_count == 0 and numerical_status == "PASS" else "FAIL"
            ),
            "comparison": "host_int8_prototype_vs_actual_e84_not_scientific_validation",
            "compared_results": len(compared),
            "output_quantum_tolerance": numerical_tolerance,
            "numerical_status": numerical_status,
            "member_maximum_absolute_error": member_maximum,
            "ensemble_maximum_absolute_error": ensemble_maximum,
            "threshold_crossing_mismatches": crossing_mismatches,
            "persistence_count_mismatches": count_mismatches,
            "reflex_required_mismatches": reflex_mismatches,
            "reflex_onset_mismatches": onset_mismatches,
        }


def replay_packets(
    serial_port: Any,
    packets: Iterable[Packet],
    rate_hz: float,
    response_timeout_s: float,
    expected: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Send a full-duplex paced replay over an already-open serial object."""
    if rate_hz <= 0.0:
        raise ValueError("replay rate must be positive")
    packet_list = list(packets)
    reset_packet: Packet | None = None
    if packet_list and packet_list[0].payload_type == PayloadType.RESET_STATE:
        reset_packet = Packet(
            PayloadType.RESET_STATE,
            time.perf_counter_ns() & 0xFFFFFFFF,
            packet_list[0].window_endpoint,
        )
        packet_list[0] = reset_packet
    decoder = FrameDecoder()
    receive_times: dict[int, tuple[Packet, int]] = {}
    stop = threading.Event()
    lock = threading.Lock()

    def reader() -> None:
        while not stop.is_set():
            chunk = serial_port.read(512)
            if not chunk:
                continue
            now = time.perf_counter_ns()
            with lock:
                for response in decoder.feed(chunk):
                    if response.payload_type == PayloadType.RESULT:
                        receive_times[response.sequence_id] = (response, now)

    worker = threading.Thread(target=reader, name="fastreflex-hil-rx", daemon=True)
    worker.start()
    send_times: dict[int, int] = {}
    sender_deadline_misses = 0
    period_ns = int(1_000_000_000 / rate_hz)
    data_start_index = 0
    if reset_packet is not None:
        reset_send = time.perf_counter_ns()
        send_times[reset_packet.sequence_id] = reset_send
        serial_port.write(encode_packet(reset_packet))
        reset_deadline = time.monotonic() + response_timeout_s
        while time.monotonic() < reset_deadline:
            with lock:
                response = receive_times.get(reset_packet.sequence_id)
                if response is not None:
                    if decode_runtime_result(response[0].payload).status_code != 1:
                        stop.set()
                        worker.join(timeout=1.0)
                        raise RuntimeError("board rejected the deterministic reset")
                    break
            time.sleep(0.001)
        else:
            stop.set()
            worker.join(timeout=1.0)
            raise RuntimeError("board did not acknowledge the deterministic reset")
        data_start_index = 1
    start = time.perf_counter_ns()
    for index, packet in enumerate(packet_list[data_start_index:]):
        deadline = start + index * period_ns
        while True:
            remaining = deadline - time.perf_counter_ns()
            if remaining <= 0:
                break
            if remaining > 200_000:
                time.sleep((remaining - 100_000) / 1_000_000_000)
        now = time.perf_counter_ns()
        sender_deadline_misses += now - deadline > period_ns
        send_times[packet.sequence_id] = now
        serial_port.write(encode_packet(packet))
    wait_deadline = time.monotonic() + response_timeout_s
    while time.monotonic() < wait_deadline:
        with lock:
            if len(receive_times) >= len(packet_list):
                break
        time.sleep(0.001)
    stop.set()
    worker.join(timeout=1.0)

    evaluator = HILEvaluator(expected)
    with lock:
        for sequence in receive_times:
            if sequence in send_times:
                packet, received_ns = receive_times[sequence]
                evaluator.observe(packet, send_times[sequence], received_ns)
    report = evaluator.summary(len(packet_list), sender_deadline_misses)
    report["protocol"] = {
        "version": 1,
        "crc_errors": decoder.crc_errors,
        "discarded_bytes": decoder.discarded_bytes,
        "full_duplex": True,
    }
    return report
