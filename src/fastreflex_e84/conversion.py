"""Deterministic TFLite export and Ethos-U55 operator feasibility checks."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .handoff import DEFAULT_CONFIG, sha256_file, validate_reference_bundle
from .reference_runtime import MEMBER_SEEDS, apply_decision, load_ensemble


M2_VERDICT = "FLOAT_EXPORT_PARITY_FAIL_INT8_U55_OPERATOR_MAPPING_PASS"
M21_VERDICT = "FLOAT_EXPORT_NUMERICAL_CONTRACT_RESOLVED"
M3_VERDICT = "INT8_DECISION_PARITY_PASS_NUMERICAL_CONTRACT_FAIL"
STATE_KEYS = (
    "gru.weight_ih_l0",
    "gru.weight_hh_l0",
    "gru.bias_ih_l0",
    "gru.bias_hh_l0",
    "classifier.weight",
    "classifier.bias",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolve_output(repository_root: Path, configured: str) -> Path:
    root = repository_root.resolve()
    output = (root / configured).resolve()
    _require(root in output.parents, "generated output path escapes the repository")
    return output


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required M2 package is not installed: {name}") from exc


def _tensorflow() -> Any:
    # Conversion is deliberately CPU-only. TensorFlow may still report that its
    # binary contains CUDA registrations, but it must not select a GPU backend.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for M2 TFLite export") from exc
    return tf


def _litert() -> tuple[Any, Any]:
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
    except ImportError as exc:
        raise RuntimeError("ai-edge-litert is required for M2 parity") from exc
    return Interpreter, OpResolverType


def _state_arrays(model: torch.nn.Module) -> dict[str, np.ndarray]:
    state = model.state_dict()
    _require(tuple(state) == STATE_KEYS, "frozen checkpoint state order changed")
    return {
        name: state[name].detach().cpu().numpy().astype(np.float32, copy=True)
        for name in STATE_KEYS
    }


def _concrete_gru(state: dict[str, np.ndarray], include_softmax: bool) -> Any:
    """Lower the exact PyTorch reset-after GRU equation to static primitives."""
    tf = _tensorflow()

    class FrozenMember(tf.Module):
        def __init__(self) -> None:
            super().__init__()
            for name, value in state.items():
                setattr(self, name.replace(".", "_"), tf.constant(value))

        @tf.function(
            input_signature=[tf.TensorSpec([1, 20, 80], tf.float32, name="window")],
            autograph=False,
        )
        def infer(self, window: Any) -> dict[str, Any]:
            hidden = tf.zeros([1, 32], tf.float32)
            # PyTorch projects all timesteps through W_ih as one contiguous
            # linear call before the recurrent loop. Keep the same batching so
            # the target Float accumulation follows the source backend closely.
            input_gates_sequence = (
                tf.matmul(
                    tf.reshape(window, [20, 80]),
                    self.gru_weight_ih_l0,
                    transpose_b=True,
                )
                + self.gru_bias_ih_l0
            )
            input_reset_sequence, input_update_sequence, input_new_sequence = tf.split(
                input_gates_sequence, 3, axis=1
            )
            for timestep in range(20):
                hidden_gates = (
                    tf.matmul(
                        hidden,
                        self.gru_weight_hh_l0,
                        transpose_b=True,
                    )
                    + self.gru_bias_hh_l0
                )
                input_reset = input_reset_sequence[timestep : timestep + 1]
                input_update = input_update_sequence[timestep : timestep + 1]
                input_new = input_new_sequence[timestep : timestep + 1]
                hidden_reset, hidden_update, hidden_new = tf.split(
                    hidden_gates, 3, axis=1
                )
                reset = tf.sigmoid(input_reset + hidden_reset)
                update = tf.sigmoid(input_update + hidden_update)
                new = tf.tanh(input_new + reset * hidden_new)
                # This is PyTorch's operation order: n + z * (h - n).
                hidden = new + update * (hidden - new)
            logits = (
                tf.matmul(hidden, self.classifier_weight, transpose_b=True)
                + self.classifier_bias
            )
            if include_softmax:
                return {"probabilities": tf.nn.softmax(logits, axis=1)}
            return {"logits": logits}

    module = FrozenMember()
    return module, module.infer.get_concrete_function()


def _float_tflite(state: dict[str, np.ndarray]) -> bytes:
    tf = _tensorflow()
    module, concrete = _concrete_gru(state, include_softmax=False)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return bytes(converter.convert())


def _int8_operator_probe(
    state: dict[str, np.ndarray], representative_windows: np.ndarray
) -> bytes:
    """Make the minimum INT8 graph needed to ask Vela about all model ops.

    This artifact is an operator probe, not an INT8 parity or accuracy result.
    Softmax is included so that M2 does not infer its mapping from documentation.
    """
    return _int8_tflite(state, representative_windows, include_softmax=True)


def _int8_tflite(
    state: dict[str, np.ndarray],
    representative_windows: np.ndarray,
    *,
    include_softmax: bool,
) -> bytes:
    """Fully quantize one frozen member using explicit INT8 graph IO."""
    tf = _tensorflow()
    module, concrete = _concrete_gru(state, include_softmax=include_softmax)

    def representative_dataset() -> Any:
        for window in representative_windows:
            yield [window[None, ...]]

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return bytes(converter.convert())


def _interpreter(model_path: Path) -> Any:
    Interpreter, OpResolverType = _litert()
    interpreter = Interpreter(
        model_path=str(model_path),
        experimental_op_resolver_type=(
            OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    interpreter.allocate_tensors()
    return interpreter


def _tensor_contract(detail: dict[str, Any]) -> dict[str, object]:
    scale, zero_point = detail["quantization"]
    return {
        "name": str(detail["name"]),
        "shape": [int(value) for value in detail["shape"]],
        "dtype": np.dtype(detail["dtype"]).name,
        "quantization": {
            "scale": float(scale),
            "zero_point": int(zero_point),
        },
    }


def inspect_tflite(model_path: Path) -> dict[str, object]:
    interpreter = _interpreter(model_path)
    operations = [str(record["op_name"]) for record in interpreter._get_ops_details()]
    return {
        "inputs": [
            _tensor_contract(detail) for detail in interpreter.get_input_details()
        ],
        "outputs": [
            _tensor_contract(detail) for detail in interpreter.get_output_details()
        ],
        "operator_count": len(operations),
        "operators": dict(sorted(Counter(operations).items())),
        "static_shapes": True,
    }


def _quantization_summary(model_path: Path) -> dict[str, object]:
    """Summarize weight, bias, activation, and recurrent quantization behavior."""
    interpreter = _interpreter(model_path)
    inputs = {int(item["index"]) for item in interpreter.get_input_details()}
    outputs = {int(item["index"]) for item in interpreter.get_output_details()}
    constants: list[dict[str, object]] = []
    biases: list[dict[str, object]] = []
    activations: list[dict[str, object]] = []
    recurrent_scales: list[float] = []
    for detail in interpreter.get_tensor_details():
        parameters = detail["quantization_parameters"]
        scales = np.asarray(parameters["scales"], dtype=np.float64)
        if not len(scales):
            continue
        record = {
            "index": int(detail["index"]),
            "name": str(detail["name"]),
            "shape": [int(value) for value in detail["shape"]],
            "dtype": np.dtype(detail["dtype"]).name,
            "scheme": "per_axis" if len(scales) > 1 else "per_tensor",
            "scale_count": len(scales),
            "quantized_dimension": int(parameters["quantized_dimension"]),
            "scale_minimum": float(np.min(scales)),
            "scale_maximum": float(np.max(scales)),
        }
        is_constant = str(detail["name"]).startswith("tfl.pseudo_qconst")
        if is_constant and np.dtype(detail["dtype"]) == np.dtype(np.int8):
            constants.append(record)
        elif is_constant and np.dtype(detail["dtype"]) == np.dtype(np.int32):
            biases.append(record)
        else:
            activations.append(record)
            if re.match(r"^MatMul_\d+;add_\d+$", str(detail["name"])):
                recurrent_scales.append(float(scales[0]))
    return {
        "quantized_tensor_count": len(constants) + len(biases) + len(activations),
        "int8_constant_tensor_count": len(constants),
        "learned_matrix_weight_tensors": [
            row for row in constants if row["scheme"] == "per_axis"
        ],
        "other_int8_constant_tensor_count": sum(
            row["scheme"] == "per_tensor" for row in constants
        ),
        "bias_tensor_count": len(biases),
        "bias_per_axis_count": sum(row["scheme"] == "per_axis" for row in biases),
        "activation_tensor_count": len(activations),
        "activation_per_axis_count": sum(
            row["scheme"] == "per_axis" for row in activations
        ),
        "activation_per_tensor_scale": {
            "minimum": min(float(row["scale_minimum"]) for row in activations),
            "maximum": max(float(row["scale_maximum"]) for row in activations),
        },
        "input_and_output_are_per_tensor": all(
            row["scheme"] == "per_tensor"
            for row in activations
            if row["index"] in inputs | outputs
        ),
        "recurrent_intermediate_scale": {
            "count": len(recurrent_scales),
            "minimum": min(recurrent_scales),
            "maximum": max(recurrent_scales),
            "by_timestep": recurrent_scales,
        },
    }


def _error_distribution(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    error = np.abs(
        actual.astype(np.float64, copy=False) - expected.astype(np.float64, copy=False)
    )
    return {
        "maximum_absolute_error": float(np.max(error)),
        "mean_absolute_error": float(np.mean(error)),
        "p50_absolute_error": float(np.percentile(error, 50)),
        "p90_absolute_error": float(np.percentile(error, 90)),
        "p95_absolute_error": float(np.percentile(error, 95)),
        "p99_absolute_error": float(np.percentile(error, 99)),
        "signed_bias": float(np.mean(actual.astype(np.float64) - expected)),
        "maximum_error_index": int(np.argmax(error)),
    }


def _run_int8_model(
    model_path: Path, windows: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    interpreter = _interpreter(model_path)
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    _require(len(inputs) == len(outputs) == 1, "unexpected INT8 TFLite IO count")
    _require(
        list(inputs[0]["shape"]) == [1, 20, 80]
        and inputs[0]["dtype"] == np.int8,
        "formal INT8 input contract changed",
    )
    _require(
        list(outputs[0]["shape"]) == [1, 2]
        and outputs[0]["dtype"] == np.int8,
        "formal INT8 output contract changed",
    )
    input_scale, input_zero = inputs[0]["quantization"]
    output_scale, output_zero = outputs[0]["quantization"]
    _require(input_scale > 0.0 and output_scale > 0.0, "INT8 IO scale is invalid")
    values: list[np.ndarray] = []
    input_error: list[np.ndarray] = []
    saturation_count = 0
    for window in windows:
        unbounded = np.rint(window / input_scale + input_zero)
        saturation_count += int(np.count_nonzero((unbounded < -128) | (unbounded > 127)))
        quantized = np.clip(unbounded, -128, 127).astype(np.int8)
        dequantized_input = (quantized.astype(np.float32) - input_zero) * input_scale
        input_error.append(np.abs(dequantized_input - window))
        interpreter.set_tensor(inputs[0]["index"], quantized[None, ...])
        interpreter.invoke()
        quantized_output = interpreter.get_tensor(outputs[0]["index"])[0].copy()
        values.append(
            (quantized_output.astype(np.float32) - output_zero) * output_scale
        )
    all_input_error = np.concatenate([value.reshape(-1) for value in input_error])
    input_elements = int(windows.size)
    return np.asarray(values, dtype=np.float32), {
        "input": _tensor_contract(inputs[0]),
        "output": _tensor_contract(outputs[0]),
        "input_quantization": {
            "formula": "clip(round(float_value / scale + zero_point), -128, 127)",
            "element_count": input_elements,
            "saturation_count": saturation_count,
            "saturation_fraction": saturation_count / input_elements,
            "maximum_absolute_dequantization_error": float(np.max(all_input_error)),
            "p99_absolute_dequantization_error": float(
                np.percentile(all_input_error, 99)
            ),
        },
    }


def _run_float_model(model_path: Path, windows: np.ndarray) -> np.ndarray:
    interpreter = _interpreter(model_path)
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    _require(len(inputs) == len(outputs) == 1, "unexpected TFLite IO count")
    _require(
        list(inputs[0]["shape"]) == [1, 20, 80] and inputs[0]["dtype"] == np.float32,
        "Float TFLite input contract changed",
    )
    _require(
        list(outputs[0]["shape"]) == [1, 2] and outputs[0]["dtype"] == np.float32,
        "Float TFLite output contract changed",
    )
    actual: list[np.ndarray] = []
    for window in windows:
        interpreter.set_tensor(inputs[0]["index"], window[None, ...])
        interpreter.invoke()
        actual.append(interpreter.get_tensor(outputs[0]["index"])[0].copy())
    return np.asarray(actual, dtype=np.float32)


def _numeric_parity(
    actual: np.ndarray, expected: np.ndarray, absolute: float, relative: float
) -> dict[str, object]:
    _require(
        actual.shape == expected.shape and actual.dtype == expected.dtype,
        "conversion parity shape or dtype changed",
    )
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    maximum = 0.0 if not difference.size else float(np.max(difference))
    mean = 0.0 if not difference.size else float(np.mean(difference))
    permitted = absolute + relative * np.abs(expected.astype(np.float64))
    violations = int(np.count_nonzero(difference > permitted))
    return {
        "status": "PASS" if violations == 0 else "FAIL",
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_absolute_error": maximum,
        "mean_absolute_error": mean,
        "tolerance_violation_count": violations,
    }


def _exact_parity(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    _require(
        actual.shape == expected.shape and actual.dtype == expected.dtype,
        "converted decision shape or dtype changed",
    )
    exact = bool(np.array_equal(actual, expected))
    return {
        "status": "PASS" if exact else "FAIL",
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "exact": exact,
        "mismatch_count": int(np.count_nonzero(actual != expected)),
    }


def _softmax_hazard(logits: np.ndarray) -> np.ndarray:
    """Float32 member softmax followed by the frozen float64 promotion."""
    shifted = (logits - np.max(logits, axis=2, keepdims=True)).astype(
        np.float32, copy=False
    )
    exponent = np.exp(shifted).astype(np.float32, copy=False)
    denominator = np.sum(exponent, axis=2, dtype=np.float32)
    return (exponent[:, :, 1] / denominator).astype(np.float32).astype(np.float64)


def _write_artifact(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _match_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return None if match is None else int(match.group(1))


def _match_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return None if match is None else float(match.group(1))


def _vela_summary(text: str) -> dict[str, object]:
    cpu_types = Counter(re.findall(r"^\s+CPU: (\w+) =", text, re.MULTILINE))
    npu_types = Counter(re.findall(r"^\s+NPU: (\w+) =", text, re.MULTILINE))
    return {
        "cpu_operators": _match_int(r"CPU operators =\s+(\d+)", text),
        "cpu_percent": _match_float(r"CPU operators =\s+\d+\s+\(([0-9.]+)%\)", text),
        "npu_operators": _match_int(r"NPU operators =\s+(\d+)", text),
        "npu_percent": _match_float(r"NPU operators =\s+\d+\s+\(([0-9.]+)%\)", text),
        "cpu_operator_types": dict(sorted(cpu_types.items())),
        "npu_operator_types": dict(sorted(npu_types.items())),
        "sram_kib": _match_float(r"Total SRAM used\s+([0-9.]+) KiB", text),
        "on_chip_flash_kib": _match_float(
            r"Total On-chip Flash used\s+([0-9.]+) KiB", text
        ),
        "off_chip_flash_kib": _match_float(
            r"Total Off-chip Flash used\s+([0-9.]+) KiB", text
        ),
        "macs_per_inference": _match_int(
            r"Neural network macs\s+(\d+) MACs/batch", text
        ),
        "npu_cycles": _match_int(r"NPU cycles\s+(\d+) cycles/batch", text),
        "total_cycles": _match_int(r"Total cycles\s+(\d+) cycles/batch", text),
        "tool_estimated_inference_ms": _match_float(
            r"Batch Inference time\s+([0-9.]+) ms", text
        ),
        "maximum_subgraph_kib": _match_float(
            r"Maximum NNG Subgraph Size = ([0-9.]+) KiB", text
        ),
        "unsupported_semantics_warnings": len(
            re.findall(
                r"^Warning: Unsupported TensorFlow Lite semantics", text, re.MULTILINE
            )
        ),
    }


def _run_vela(
    vela: str,
    model_path: Path,
    output_directory: Path,
    accelerator: str,
    system_config: str,
    memory_mode: str,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    command = [
        vela,
        str(model_path),
        "--config",
        "Arm/vela.ini",
        "--accelerator-config",
        accelerator,
        "--system-config",
        system_config,
        "--memory-mode",
        memory_mode,
        "--output-dir",
        str(output_directory),
        "--show-cpu-operations",
        "--show-subgraph-io-summary",
        "--verbose-cycle-estimate",
        "--verbose-weights",
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Vela failed for {model_path.name}/{memory_mode}:\n{completed.stdout}"
        )
    log_path = output_directory / "vela.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    compiled = output_directory / f"{model_path.stem}_vela.tflite"
    _require(compiled.is_file(), "Vela did not create the expected TFLite output")
    return {
        "command": command,
        "memory_mode": memory_mode,
        "log_path": str(log_path),
        "compiled_model": {
            "path": str(compiled),
            "bytes": compiled.stat().st_size,
            "sha256": sha256_file(compiled),
        },
        **_vela_summary(completed.stdout),
    }


def evaluate_export_feasibility(
    repository_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path | None = None,
) -> dict[str, object]:
    """Export all frozen members, prove Float parity, and query actual Vela."""
    root = repository_root.resolve()
    bundle, config, model_manifest, float_contract = validate_reference_bundle(
        root, config_path
    )
    settings = config.get("export_feasibility")
    _require(isinstance(settings, dict), "export feasibility config is missing")
    seeds = [int(value) for value in settings["member_seeds"]]
    _require(seeds == list(MEMBER_SEEDS), "export member order changed")
    probe_seed = int(settings["operator_probe_seed"])
    _require(probe_seed in seeds, "operator probe seed is not an ensemble member")

    if output_root is None:
        float_directory = _resolve_output(root, settings["float_directory"])
        quantized_directory = _resolve_output(root, settings["quantized_directory"])
        vela_directory = _resolve_output(root, settings["vela_directory"])
        result_path = _resolve_output(root, settings["result_path"])
    else:
        selected_root = output_root.resolve()
        float_directory = selected_root / "converted"
        quantized_directory = selected_root / "quantized"
        vela_directory = selected_root / "vela"
        result_path = selected_root / "export_target_feasibility.json"

    with np.load(
        bundle / "golden_outputs/deployment_runtime_chain.npz", allow_pickle=False
    ) as golden:
        windows = golden["model_windows"].copy()
        expected_logits = golden["member_logits"].copy()
        expected_probabilities = golden["member_hazard_probability"].copy()
        expected_ensemble = golden["ensemble_hazard_probability"].copy()
        expected_crossing = golden["threshold_crossing"].copy()
        expected_counts = golden["consecutive_threshold_count"].copy()
        expected_reflex = golden["reflex_required"].copy()
        expected_onset = golden["reflex_onset"].copy()
    _require(windows.shape == (121, 20, 80), "golden model windows changed")

    models = load_ensemble(bundle, model_manifest)
    float_artifacts: list[dict[str, object]] = []
    converted_logits: list[np.ndarray] = []
    float_graph: dict[str, object] | None = None
    probe_state: dict[str, np.ndarray] | None = None
    for seed, model in zip(seeds, models):
        state = _state_arrays(model)
        output = float_directory / f"member_seed{seed}_float32.tflite"
        artifact = _write_artifact(output, _float_tflite(state))
        graph = inspect_tflite(output)
        expected_graph = settings["expected_float_operators"]
        _require(graph["operators"] == expected_graph, "Float TFLite graph changed")
        actual = _run_float_model(output, windows)
        converted_logits.append(actual)
        artifact.update({"seed": seed, "graph": graph})
        float_artifacts.append(artifact)
        float_graph = graph
        if seed == probe_seed:
            probe_state = state
    _require(probe_state is not None and float_graph is not None, "probe state missing")

    actual_logits = np.stack(converted_logits).astype(np.float32, copy=False)
    actual_probabilities = _softmax_hazard(actual_logits)
    actual_ensemble = np.mean(actual_probabilities, axis=0, dtype=np.float64)
    crossing, counts, reflex, onset = apply_decision(actual_ensemble)
    tolerance = float_contract["continuous_parity"]
    logit_absolute = float(tolerance["member_logits"]["absolute"])
    logit_relative = float(tolerance["member_logits"]["relative"])
    probability_absolute = float(tolerance["member_hazard_probability"]["absolute"])
    probability_relative = float(tolerance["member_hazard_probability"]["relative"])
    ensemble_absolute = float(tolerance["ensemble_hazard_probability"]["absolute"])
    ensemble_relative = float(tolerance["ensemble_hazard_probability"]["relative"])
    member_parity = {
        str(seed): _numeric_parity(
            actual_logits[index],
            expected_logits[index],
            logit_absolute,
            logit_relative,
        )
        for index, seed in enumerate(seeds)
    }
    parity = {
        "tolerance": tolerance,
        "member_logits": _numeric_parity(
            actual_logits,
            expected_logits,
            logit_absolute,
            logit_relative,
        ),
        "member_logits_by_seed": member_parity,
        "member_hazard_probability": _numeric_parity(
            actual_probabilities,
            expected_probabilities,
            probability_absolute,
            probability_relative,
        ),
        "ensemble_hazard_probability": _numeric_parity(
            actual_ensemble,
            expected_ensemble,
            ensemble_absolute,
            ensemble_relative,
        ),
        "threshold_crossing": _exact_parity(crossing, expected_crossing),
        "consecutive_threshold_count": _exact_parity(counts, expected_counts),
        "reflex_required": _exact_parity(reflex, expected_reflex),
        "reflex_onset": _exact_parity(onset, expected_onset),
    }
    parity["status"] = (
        "PASS"
        if all(
            record["status"] == "PASS"
            for name, record in parity.items()
            if name not in {"tolerance", "member_logits_by_seed"}
        )
        else "FAIL"
    )

    probe_path = (
        quantized_directory / f"member_seed{probe_seed}_int8_operator_probe.tflite"
    )
    probe_artifact = _write_artifact(
        probe_path, _int8_operator_probe(probe_state, windows)
    )
    probe_graph = inspect_tflite(probe_path)
    _require(
        probe_graph["operators"] == settings["expected_int8_probe_operators"],
        "INT8 operator-probe graph changed",
    )
    probe_artifact.update(
        {
            "seed": probe_seed,
            "purpose": "operator_mapping_only_not_int8_parity",
            "representative_source": "non-protected M1 golden V2_VALIDATION windows",
            "graph": probe_graph,
        }
    )

    vela = shutil.which("vela")
    if vela is None:
        raise RuntimeError("Vela executable is not installed or not on PATH")
    target = settings["target"]
    accelerator = str(target["accelerator_config"])
    system_config = str(target["generic_system_config"])
    float_vela = _run_vela(
        vela,
        Path(float_artifacts[seeds.index(probe_seed)]["path"]),
        vela_directory / "float_shared_sram",
        accelerator,
        system_config,
        "Shared_Sram",
    )
    int8_shared = _run_vela(
        vela,
        probe_path,
        vela_directory / "int8_shared_sram",
        accelerator,
        system_config,
        "Shared_Sram",
    )
    int8_sram_only = _run_vela(
        vela,
        probe_path,
        vela_directory / "int8_sram_only",
        accelerator,
        system_config,
        "Sram_Only",
    )
    _require(
        float_vela["cpu_operators"] == float_graph["operator_count"]
        and float_vela["npu_operators"] == 0,
        "Float graph unexpectedly mapped to the U55",
    )
    _require(
        int8_shared["cpu_operators"] == 0
        and int8_shared["npu_operators"] > 0
        and int8_sram_only["cpu_operators"] == 0
        and int8_sram_only["npu_operators"] > 0,
        "quantized operator probe did not map completely to the U55",
    )

    target_clock = int(target["device_core_clock_hz"])
    total_cycles = int8_sram_only["total_cycles"]
    _require(isinstance(total_cycles, int), "Vela did not report cycle estimates")
    analytical_member_us = total_cycles / target_clock * 1_000_000.0
    float_contract_passed = parity["status"] == "PASS"
    result: dict[str, object] = {
        "status": (
            M21_VERDICT
            if float_contract_passed
            else "FLOAT_EXPORT_NUMERICAL_CONTRACT_REVALIDATION_FAIL"
        ),
        "reference": {
            "candidate_id": model_manifest["candidate_id"],
            "role": model_manifest["engineering_role"],
            "scientific_verdict": model_manifest["scientific_status"]["verdict"],
            "research_release_commit": config["accepted_identity"][
                "research_release_commit"
            ],
            "release_manifest_sha256": config["accepted_identity"][
                "release_manifest_sha256"
            ],
        },
        "toolchain": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tensorflow": _package_version("tensorflow"),
            "ai_edge_litert": _package_version("ai-edge-litert"),
            "ethos_u_vela": _package_version("ethos-u-vela"),
            "vela_executable": vela,
        },
        "float_export": {
            "format": "TFLite built-ins, float32, static batch 1",
            "lowering": "PyTorch reset-after GRU equation statically unrolled for 20 steps",
            "members": float_artifacts,
            "parity": parity,
        },
        "int8_operator_probe": probe_artifact,
        "vela": {
            "accelerator_config": accelerator,
            "generic_system_config": system_config,
            "configuration_is_e84_specific": False,
            "float_shared_sram": float_vela,
            "int8_shared_sram": int8_shared,
            "int8_sram_only": int8_sram_only,
        },
        "runtime_feasibility": {
            "member_macs": int8_sram_only["macs_per_inference"],
            "three_member_macs": 3 * int(int8_sram_only["macs_per_inference"]),
            "vela_sram_only_cycles_per_member": total_cycles,
            "analytical_sram_only_us_per_member_at_400mhz": analytical_member_us,
            "analytical_sram_only_us_for_three_at_400mhz": (3.0 * analytical_member_us),
            "deadline_us": 1000,
            "measured_on_board": False,
            "qualification": (
                "the Research-owned batch-one Float contract passes and operator "
                "mapping remains viable; target-specific memory configuration, "
                "formal INT8 parity, preprocessing, invocation overhead, and "
                "board timing remain"
                if float_contract_passed
                else "the Research-owned batch-one Float contract did not pass; "
                "formal INT8 work remains blocked"
            ),
        },
        "boundary": {
            "int8_parity_completed": False,
            "m3_authorized": float_contract_passed,
            "firmware_started": False,
            "board_state_modified": False,
            "research_semantics_modified": False,
        },
        "next_milestone": (
            "INT8_QUANTIZATION_AND_PARITY"
            if float_contract_passed
            else "FLOAT_EXPORT_NUMERICAL_CONTRACT_RESOLUTION"
        ),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["result_path"] = str(result_path)
    return result


def _evaluate_int8_ensemble(
    *,
    states: list[dict[str, np.ndarray]],
    seeds: list[int],
    calibration: np.ndarray,
    windows: np.ndarray,
    golden: dict[str, np.ndarray],
    output_directory: Path,
    include_softmax: bool,
    expected_operators: dict[str, int],
    repeat_conversion: bool,
) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    dequantized_outputs: list[np.ndarray] = []
    suffix = "probability" if include_softmax else "logits"
    for seed, state in zip(seeds, states):
        payload = _int8_tflite(
            state, calibration, include_softmax=include_softmax
        )
        path = output_directory / f"member_seed{seed}_int8_{suffix}.tflite"
        artifact = _write_artifact(path, payload)
        graph = inspect_tflite(path)
        _require(
            graph["operators"] == expected_operators,
            f"formal INT8 {suffix} graph changed for seed {seed}",
        )
        values, io = _run_int8_model(path, windows)
        deterministic = True
        if repeat_conversion:
            deterministic = (
                _int8_tflite(
                    state, calibration, include_softmax=include_softmax
                )
                == payload
            )
            _require(deterministic, f"INT8 conversion is not deterministic: {seed}")
        artifact.update(
            {
                "seed": seed,
                "graph": graph,
                "io": io,
                "quantization_summary": _quantization_summary(path),
                "repeated_conversion_byte_identical": deterministic,
            }
        )
        artifacts.append(artifact)
        dequantized_outputs.append(values)

    stacked = np.stack(dequantized_outputs).astype(np.float32, copy=False)
    if include_softmax:
        logits: np.ndarray | None = None
        probabilities = stacked[:, :, 1].astype(np.float64)
        output_semantics = "dequantized_softmax_class_1_probability"
    else:
        logits = stacked
        probabilities = _softmax_hazard(logits)
        output_semantics = "dequantized_logits_then_host_float32_softmax_class_1"
    ensemble = np.mean(probabilities, axis=0, dtype=np.float64)
    crossing, counts, reflex, onset = apply_decision(ensemble)
    probability_by_seed: dict[str, object] = {}
    for index, seed in enumerate(seeds):
        distribution = _error_distribution(
            probabilities[index], golden["member_hazard_probability"][index]
        )
        error_index = int(distribution["maximum_error_index"])
        distribution["float_at_maximum_error"] = float(
            golden["member_hazard_probability"][index, error_index]
        )
        distribution["int8_at_maximum_error"] = float(
            probabilities[index, error_index]
        )
        probability_by_seed[str(seed)] = distribution
    parity: dict[str, object] = {
        "member_hazard_probability": _error_distribution(
            probabilities, golden["member_hazard_probability"]
        ),
        "member_hazard_probability_by_seed": probability_by_seed,
        "ensemble_hazard_probability": _error_distribution(
            ensemble, golden["ensemble_hazard_probability"]
        ),
        "threshold_crossing": _exact_parity(
            crossing, golden["threshold_crossing"]
        ),
        "consecutive_threshold_count": _exact_parity(
            counts, golden["consecutive_threshold_count"]
        ),
        "reflex_required": _exact_parity(reflex, golden["reflex_required"]),
        "reflex_onset": _exact_parity(onset, golden["reflex_onset"]),
    }
    if logits is not None:
        parity["member_logits"] = _error_distribution(
            logits, golden["member_logits"]
        )
    return {
        "representation": output_semantics,
        "members": artifacts,
        "parity": parity,
        "actual": {
            "probabilities": probabilities,
            "ensemble": ensemble,
            "threshold_crossing": crossing,
            "consecutive_threshold_count": counts,
            "reflex_required": reflex,
            "reflex_onset": onset,
        },
    }


def evaluate_int8_quantization(
    repository_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path | None = None,
) -> dict[str, object]:
    """Quantize all members, characterize parity, and compile the formal graphs."""
    root = repository_root.resolve()
    bundle, config, model_manifest, _ = validate_reference_bundle(root, config_path)
    settings = config.get("formal_int8")
    _require(isinstance(settings, dict), "formal INT8 config is missing")
    seeds = [int(value) for value in settings["member_seeds"]]
    _require(seeds == list(MEMBER_SEEDS), "formal INT8 member order changed")

    calibration_manifest = json.loads(
        (bundle / "calibration_manifest.json").read_text(encoding="utf-8")
    )
    with np.load(
        bundle / "calibration_inputs/int8_representative.npz", allow_pickle=False
    ) as payload:
        representative = payload["model_windows"].copy()
    calibration_policy = settings["calibration_policy"]
    percentile = float(calibration_policy["symmetric_absolute_percentile"])
    _require(percentile == 99.0, "reviewed INT8 calibration policy changed")
    clipping_bound = float(np.percentile(np.abs(representative), percentile))
    robust_calibration = np.clip(
        representative, -clipping_bound, clipping_bound
    ).astype(np.float32, copy=False)

    with np.load(
        bundle / "golden_outputs/deployment_runtime_chain.npz", allow_pickle=False
    ) as payload:
        golden = {name: payload[name].copy() for name in payload.files}
    windows = golden["model_windows"]
    _require(windows.shape == (121, 20, 80), "deployment golden windows changed")
    models = load_ensemble(bundle, model_manifest)
    states = [_state_arrays(model) for model in models]

    if output_root is None:
        selected_directory = _resolve_output(root, settings["selected_directory"])
        alternative_directory = _resolve_output(
            root, settings["alternative_directory"]
        )
        vela_directory = _resolve_output(root, settings["vela_directory"])
        result_path = _resolve_output(root, settings["result_path"])
    else:
        selected_root = output_root.resolve()
        selected_directory = selected_root / "quantized/selected_npu_softmax"
        alternative_directory = selected_root / "quantized/alternatives"
        vela_directory = selected_root / "vela"
        result_path = selected_root / "int8_quantization_parity.json"

    selected = _evaluate_int8_ensemble(
        states=states,
        seeds=seeds,
        calibration=robust_calibration,
        windows=windows,
        golden=golden,
        output_directory=selected_directory,
        include_softmax=True,
        expected_operators=settings["expected_probability_operators"],
        repeat_conversion=True,
    )
    full_range = _evaluate_int8_ensemble(
        states=states,
        seeds=seeds,
        calibration=representative,
        windows=windows,
        golden=golden,
        output_directory=alternative_directory / "full_range_npu_softmax",
        include_softmax=True,
        expected_operators=settings["expected_probability_operators"],
        repeat_conversion=False,
    )
    host_softmax = _evaluate_int8_ensemble(
        states=states,
        seeds=seeds,
        calibration=robust_calibration,
        windows=windows,
        golden=golden,
        output_directory=alternative_directory / "int8_logits_host_softmax",
        include_softmax=False,
        expected_operators=settings["expected_logit_operators"],
        repeat_conversion=False,
    )

    contract_settings = settings["numerical_contract"]
    member_records = selected["parity"]["member_hazard_probability_by_seed"]
    ensemble_record = selected["parity"]["ensemble_hazard_probability"]
    input_record = selected["members"][0]["io"]["input_quantization"]
    continuous_checks = {
        "input_saturation_fraction": {
            "observed": input_record["saturation_fraction"],
            "maximum": float(contract_settings["input_saturation_fraction_maximum"]),
        },
        "member_probability_median_absolute_error": {
            "observed_maximum_across_members": max(
                float(record["p50_absolute_error"])
                for record in member_records.values()
            ),
            "maximum": float(
                contract_settings["member_probability_median_absolute_error_maximum"]
            ),
        },
        "member_probability_maximum_absolute_error": {
            "observed_maximum_across_members": max(
                float(record["maximum_absolute_error"])
                for record in member_records.values()
            ),
            "maximum": float(
                contract_settings["member_probability_maximum_absolute_error_maximum"]
            ),
        },
        "ensemble_probability_p95_absolute_error": {
            "observed": ensemble_record["p95_absolute_error"],
            "maximum": float(
                contract_settings["ensemble_probability_p95_absolute_error_maximum"]
            ),
        },
        "ensemble_probability_maximum_absolute_error": {
            "observed": ensemble_record["maximum_absolute_error"],
            "maximum": float(
                contract_settings["ensemble_probability_maximum_absolute_error_maximum"]
            ),
        },
        "ensemble_probability_absolute_bias": {
            "observed": abs(float(ensemble_record["signed_bias"])),
            "maximum": float(
                contract_settings["ensemble_probability_absolute_bias_maximum"]
            ),
        },
    }
    for record in continuous_checks.values():
        observed = float(
            record.get("observed", record.get("observed_maximum_across_members"))
        )
        record["status"] = "PASS" if observed <= float(record["maximum"]) else "FAIL"
    continuous_status = (
        "PASS"
        if all(record["status"] == "PASS" for record in continuous_checks.values())
        else "FAIL"
    )
    exact_names = (
        "threshold_crossing",
        "consecutive_threshold_count",
        "reflex_required",
        "reflex_onset",
    )
    exact_status = (
        "PASS"
        if all(selected["parity"][name]["status"] == "PASS" for name in exact_names)
        else "FAIL"
    )

    expected_ensemble = golden["ensemble_hazard_probability"]
    threshold = float(settings["threshold"])
    above = expected_ensemble >= threshold
    above_index = int(np.flatnonzero(above)[np.argmin(expected_ensemble[above] - threshold)])
    below_index = int(
        np.flatnonzero(~above)[np.argmin(threshold - expected_ensemble[~above])]
    )
    selected_actual = selected["actual"]
    sensitivity = {
        "threshold": threshold,
        "persistence_samples": int(settings["persistence_samples"]),
        "closest_float_above": {
            "window_index": above_index,
            "float_probability": float(expected_ensemble[above_index]),
            "int8_probability": float(selected_actual["ensemble"][above_index]),
            "float_margin": float(expected_ensemble[above_index] - threshold),
            "signed_int8_error": float(
                selected_actual["ensemble"][above_index]
                - expected_ensemble[above_index]
            ),
        },
        "closest_float_below": {
            "window_index": below_index,
            "float_probability": float(expected_ensemble[below_index]),
            "int8_probability": float(selected_actual["ensemble"][below_index]),
            "float_margin": float(threshold - expected_ensemble[below_index]),
            "signed_int8_error": float(
                selected_actual["ensemble"][below_index]
                - expected_ensemble[below_index]
            ),
        },
        "crossing_mismatch_count": selected["parity"]["threshold_crossing"][
            "mismatch_count"
        ],
        "count_mismatch_count": selected["parity"]["consecutive_threshold_count"][
            "mismatch_count"
        ],
        "reflex_mismatch_count": selected["parity"]["reflex_required"][
            "mismatch_count"
        ],
        "onset_mismatch_count": selected["parity"]["reflex_onset"][
            "mismatch_count"
        ],
        "float_onset_window_indices": np.flatnonzero(golden["reflex_onset"]).tolist(),
        "int8_onset_window_indices": np.flatnonzero(
            selected_actual["reflex_onset"]
        ).tolist(),
        "float_onset_endpoints": golden["window_endpoints"][
            golden["reflex_onset"]
        ].tolist(),
        "int8_onset_endpoints": golden["window_endpoints"][
            selected_actual["reflex_onset"]
        ].tolist(),
    }

    vela = shutil.which("vela")
    if vela is None:
        raise RuntimeError("Vela executable is not installed or not on PATH")
    target = settings["target"]
    mappings: dict[str, dict[str, object]] = {}
    for memory_mode in ("Shared_Sram", "Sram_Only"):
        by_seed: dict[str, object] = {}
        for member in selected["members"]:
            seed = int(member["seed"])
            record = _run_vela(
                vela,
                Path(member["path"]),
                vela_directory / memory_mode.lower() / f"member_seed{seed}",
                str(target["accelerator_config"]),
                str(target["generic_system_config"]),
                memory_mode,
            )
            _require(
                record["cpu_operators"] == 0 and record["npu_operators"] > 0,
                f"formal member {seed}/{memory_mode} has U55 fallback",
            )
            by_seed[str(seed)] = record
        mappings[memory_mode] = by_seed
    vela_summary: dict[str, object] = {}
    for memory_mode, by_seed in mappings.items():
        rows = list(by_seed.values())
        vela_summary[memory_mode] = {
            "three_member_macs": sum(int(row["macs_per_inference"]) for row in rows),
            "three_member_npu_cycles": sum(int(row["npu_cycles"]) for row in rows),
            "three_member_total_cycles": sum(int(row["total_cycles"]) for row in rows),
            "maximum_member_sram_kib": max(float(row["sram_kib"]) for row in rows),
            "sum_compiled_model_bytes": sum(
                int(row["compiled_model"]["bytes"]) for row in rows
            ),
            "scratch_reuse_assumption": (
                "sequential member invocation may reuse the largest member arena; "
                "TFLM arena and linker placement remain M4 measurements"
            ),
        }

    numerical_pass = continuous_status == "PASS" and exact_status == "PASS"
    status = (
        "INT8_QUANTIZATION_AND_PARITY_PASS"
        if numerical_pass
        else M3_VERDICT
    )

    def without_arrays(value: dict[str, object]) -> dict[str, object]:
        return {key: item for key, item in value.items() if key != "actual"}

    result: dict[str, object] = {
        "status": status,
        "reference": {
            "candidate_id": model_manifest["candidate_id"],
            "engineering_role": model_manifest["engineering_role"],
            "scientific_verdict": model_manifest["scientific_status"]["verdict"],
            "research_release_commit": config["accepted_identity"][
                "research_release_commit"
            ],
            "release_manifest_sha256": config["accepted_identity"][
                "release_manifest_sha256"
            ],
            "calibration_sha256": config["accepted_identity"][
                "int8_calibration_sha256"
            ],
        },
        "calibration": {
            "source_splits": calibration_manifest["source_splits"],
            "run_count": calibration_manifest["selection"]["run_count"],
            "window_count": calibration_manifest["selection"]["window_count"],
            "selection_strategy": calibration_manifest["selection"]["strategy"],
            "input_representation": calibration_manifest["representation"],
            "deployment_range_policy": {
                "method": "symmetric_clipping_before_representative_calibration",
                "absolute_percentile": percentile,
                "bound": clipping_bound,
                "fraction_of_representative_elements_clipped_by_definition": 0.01,
                "runtime_preprocessing_changed": False,
                "rationale": calibration_policy["rationale"],
            },
            "protected_holdout_access": False,
        },
        "selected_representation": {
            "name": "full_integer_int8_with_npu_softmax",
            "graph": (
                "int8 input, per-axis int8 weights, per-tensor int8 activations, "
                "NPU softmax, int8 [1,2] probabilities"
            ),
            "host_boundary": (
                "dequantize two probabilities, select class 1, promote to float64, "
                "three-member mean, >=0.99, five-sample persistence"
            ),
            "formal": without_arrays(selected),
            "selection_reason": (
                "same numerical decision result as host softmax with one fewer host "
                "nonlinear operation and complete U55 graph placement"
            ),
        },
        "alternatives": {
            "unclipped_full_train_range_npu_softmax": without_arrays(full_range),
            "robust_range_int8_logits_host_float32_softmax": without_arrays(
                host_softmax
            ),
        },
        "int8_numerical_contract": {
            "status": continuous_status,
            "checks": continuous_checks,
            "rationale": contract_settings["rationale"],
            "float_tolerances_reused": False,
            "discrete_requirement": "all_four_layers_exact",
            "discrete_status": exact_status,
        },
        "threshold_sensitivity": sensitivity,
        "vela": {
            "tool_version": _package_version("ethos-u-vela"),
            "executable": vela,
            "accelerator_config": target["accelerator_config"],
            "system_config": target["generic_system_config"],
            "configuration_is_e84_specific": False,
            "members": mappings,
            "three_member_summary": vela_summary,
            "timing_is_board_measured": False,
        },
        "toolchain": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tensorflow": _package_version("tensorflow"),
            "ai_edge_litert": _package_version("ai-edge-litert"),
            "ethos_u_vela": _package_version("ethos-u-vela"),
        },
        "boundary": {
            "m3_completed": True,
            "continuous_probability_contract_passed": continuous_status == "PASS",
            "exact_decision_parity_passed": exact_status == "PASS",
            "m4_authorized": numerical_pass,
            "firmware_started": False,
            "board_state_modified": False,
            "research_semantics_modified": False,
        },
        "next_milestone": (
            "M4_E84_FIRMWARE_AND_BOARD_EXECUTION"
            if numerical_pass
            else "INT8_RECURRENT_NUMERICAL_INSTABILITY_RESOLUTION"
        ),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["result_path"] = str(result_path)
    return result
