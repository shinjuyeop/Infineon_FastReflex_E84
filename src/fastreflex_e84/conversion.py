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
from .reference_runtime import (
    MEMBER_SEEDS,
    apply_decision,
    load_ensemble,
    load_normalizer,
)


M2_VERDICT = "FLOAT_EXPORT_PARITY_FAIL_INT8_U55_OPERATOR_MAPPING_PASS"
M21_VERDICT = "FLOAT_EXPORT_NUMERICAL_CONTRACT_RESOLVED"
M3_VERDICT = "INT8_DECISION_PARITY_PASS_NUMERICAL_CONTRACT_FAIL"
M31_VERDICT = "INT8_PTQ_PARTIAL_RECOVERY_NUMERICAL_CONTRACT_FAIL"
PROTOTYPE_ROLE = "NON_RELEASE_HIL_PATH_PROTOTYPE"
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


def _concrete_gru(
    state: dict[str, np.ndarray],
    include_softmax: bool,
    *,
    projection_block_width: int | None = None,
) -> Any:
    """Lower the exact PyTorch reset-after GRU equation to static primitives."""
    tf = _tensorflow()
    if projection_block_width is not None:
        _require(
            projection_block_width > 0 and 32 % projection_block_width == 0,
            "GRU projection block width must divide the frozen hidden size",
        )

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
            flattened = tf.reshape(window, [20, 80])
            if projection_block_width is None:
                # PyTorch projects all timesteps through W_ih as one contiguous
                # linear call before the recurrent loop. Keep the same batching so
                # the target Float accumulation follows the source backend closely.
                input_gates_sequence = (
                    tf.matmul(
                        flattened,
                        self.gru_weight_ih_l0,
                        transpose_b=True,
                    )
                    + self.gru_bias_ih_l0
                )
                (
                    input_reset_sequence,
                    input_update_sequence,
                    input_new_sequence,
                ) = tf.split(input_gates_sequence, 3, axis=1)
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
            else:
                blocks_per_gate = 32 // projection_block_width
                block_count = 3 * blocks_per_gate
                input_weights = tf.split(self.gru_weight_ih_l0, block_count, axis=0)
                input_biases = tf.split(self.gru_bias_ih_l0, block_count)
                hidden_weights = tf.split(self.gru_weight_hh_l0, block_count, axis=0)
                hidden_biases = tf.split(self.gru_bias_hh_l0, block_count)
                input_blocks = [
                    tf.matmul(flattened, weight, transpose_b=True) + bias
                    for weight, bias in zip(input_weights, input_biases)
                ]
                for timestep in range(20):
                    hidden_blocks = [
                        tf.matmul(hidden, weight, transpose_b=True) + bias
                        for weight, bias in zip(hidden_weights, hidden_biases)
                    ]
                    next_hidden = []
                    for block in range(blocks_per_gate):
                        reset = tf.sigmoid(
                            input_blocks[block][timestep : timestep + 1]
                            + hidden_blocks[block]
                        )
                        update = tf.sigmoid(
                            input_blocks[blocks_per_gate + block][
                                timestep : timestep + 1
                            ]
                            + hidden_blocks[blocks_per_gate + block]
                        )
                        new = tf.tanh(
                            input_blocks[2 * blocks_per_gate + block][
                                timestep : timestep + 1
                            ]
                            + reset * hidden_blocks[2 * blocks_per_gate + block]
                        )
                        start = block * projection_block_width
                        end = start + projection_block_width
                        previous = (
                            hidden if blocks_per_gate == 1 else hidden[:, start:end]
                        )
                        next_hidden.append(new + update * (previous - new))
                    hidden = tf.concat(next_hidden, axis=1)
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
    projection_block_width: int | None = None,
) -> bytes:
    """Fully quantize one frozen member using explicit INT8 graph IO."""
    tf = _tensorflow()
    module, concrete = _concrete_gru(
        state,
        include_softmax=include_softmax,
        projection_block_width=projection_block_width,
    )

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
        list(inputs[0]["shape"]) == [1, 20, 80] and inputs[0]["dtype"] == np.int8,
        "formal INT8 input contract changed",
    )
    _require(
        list(outputs[0]["shape"]) == [1, 2] and outputs[0]["dtype"] == np.int8,
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
        saturation_count += int(
            np.count_nonzero((unbounded < -128) | (unbounded > 127))
        )
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


def run_frozen_int8_prototype(
    repository_root: Path,
    windows: np.ndarray,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, np.ndarray]:
    """Run the frozen prototype artifacts as the host deployment oracle."""
    root = repository_root.resolve()
    _, config, _, _ = validate_reference_bundle(root, config_path)
    settings = config.get("non_release_hil_prototype")
    _require(isinstance(settings, dict), "non-release prototype config is missing")
    manifest_path = _resolve_output(root, str(settings["directory"])) / "manifest.json"
    _require(manifest_path.is_file(), "frozen prototype manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("role") == PROTOTYPE_ROLE
        and manifest.get("formal_m3_pass") is False
        and manifest.get("numerical_contract_pass") is False,
        "prototype/formal boundary changed",
    )
    values = np.asarray(windows, dtype=np.float32)
    _require(
        values.ndim == 3 and values.shape[1:] == (20, 80),
        "prototype host oracle requires [N,20,80] windows",
    )
    members: list[np.ndarray] = []
    for artifact in manifest["artifacts"]:
        model_path = root / str(artifact["path"])
        _require(
            model_path.is_file() and sha256_file(model_path) == artifact["sha256"],
            f"prototype model identity changed: {artifact['seed']}",
        )
        output, _ = _run_int8_model(model_path, values)
        members.append(output[:, 1].astype(np.float64))
    probabilities = np.stack(members)
    ensemble = np.mean(probabilities, axis=0, dtype=np.float64)
    crossing, counts, reflex, onset = apply_decision(ensemble)
    return {
        "member_hazard_probability": probabilities,
        "ensemble_hazard_probability": ensemble,
        "threshold_crossing": crossing,
        "consecutive_threshold_count": counts,
        "reflex_required": reflex,
        "reflex_onset": onset,
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


def _float_member_probabilities(
    state: dict[str, np.ndarray], windows: np.ndarray
) -> np.ndarray:
    """Run the frozen reset-after equation with Float32 PyTorch primitives."""
    values = torch.from_numpy(windows.astype(np.float32, copy=False))
    weight_ih = torch.from_numpy(state["gru.weight_ih_l0"])
    weight_hh = torch.from_numpy(state["gru.weight_hh_l0"])
    bias_ih = torch.from_numpy(state["gru.bias_ih_l0"])
    bias_hh = torch.from_numpy(state["gru.bias_hh_l0"])
    classifier_weight = torch.from_numpy(state["classifier.weight"])
    classifier_bias = torch.from_numpy(state["classifier.bias"])
    with torch.no_grad():
        projected = torch.nn.functional.linear(values, weight_ih, bias_ih)
        input_reset, input_update, input_new = projected.chunk(3, dim=2)
        hidden = torch.zeros((len(values), 32), dtype=torch.float32)
        for timestep in range(20):
            hidden_reset, hidden_update, hidden_new = torch.nn.functional.linear(
                hidden, weight_hh, bias_hh
            ).chunk(3, dim=1)
            reset = torch.sigmoid(input_reset[:, timestep] + hidden_reset)
            update = torch.sigmoid(input_update[:, timestep] + hidden_update)
            new = torch.tanh(input_new[:, timestep] + reset * hidden_new)
            hidden = new + update * (hidden - new)
        logits = torch.nn.functional.linear(hidden, classifier_weight, classifier_bias)
        probability = torch.softmax(logits, dim=1)[:, 1]
    return probability.numpy().astype(np.float64)


def _float_recurrent_trace(
    state: dict[str, np.ndarray], window: np.ndarray
) -> tuple[list[dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    """Expose the meaningful Float values in the exact static GRU equation."""
    values = torch.from_numpy(window.astype(np.float32, copy=False))
    weight_ih = torch.from_numpy(state["gru.weight_ih_l0"])
    weight_hh = torch.from_numpy(state["gru.weight_hh_l0"])
    bias_ih = torch.from_numpy(state["gru.bias_ih_l0"])
    bias_hh = torch.from_numpy(state["gru.bias_hh_l0"])
    projected = torch.nn.functional.linear(values, weight_ih, bias_ih)
    input_reset, input_update, input_new = projected.chunk(3, dim=1)
    hidden = torch.zeros((1, 32), dtype=torch.float32)
    trace: list[dict[str, np.ndarray]] = []
    with torch.no_grad():
        for timestep in range(20):
            hidden_projection = torch.nn.functional.linear(hidden, weight_hh, bias_hh)
            hidden_reset, hidden_update, hidden_new = hidden_projection.chunk(3, dim=1)
            reset_preactivation = input_reset[timestep : timestep + 1] + hidden_reset
            reset_gate = torch.sigmoid(reset_preactivation)
            reset_hidden_new = reset_gate * hidden_new
            candidate_preactivation = (
                input_new[timestep : timestep + 1] + reset_hidden_new
            )
            candidate_gate = torch.tanh(candidate_preactivation)
            hidden_difference = hidden - candidate_gate
            update_preactivation = input_update[timestep : timestep + 1] + hidden_update
            update_gate = torch.sigmoid(update_preactivation)
            update_delta = update_gate * hidden_difference
            hidden = candidate_gate + update_delta
            tensors = {
                "input_projection": projected[timestep : timestep + 1],
                "hidden_projection": hidden_projection,
                "reset_preactivation": reset_preactivation,
                "reset_gate": reset_gate,
                "reset_hidden_new": reset_hidden_new,
                "candidate_preactivation": candidate_preactivation,
                "candidate_gate": candidate_gate,
                "hidden_difference": hidden_difference,
                "update_preactivation": update_preactivation,
                "update_gate": update_gate,
                "update_delta": update_delta,
                "hidden_state": hidden,
            }
            trace.append(
                {
                    name: tensor.detach().cpu().numpy().copy()
                    for name, tensor in tensors.items()
                }
            )
        logits = torch.nn.functional.linear(
            hidden,
            torch.from_numpy(state["classifier.weight"]),
            torch.from_numpy(state["classifier.bias"]),
        )
        probabilities = torch.softmax(logits, dim=1)
    return (
        trace,
        logits.detach().cpu().numpy().copy(),
        probabilities.detach().cpu().numpy().copy(),
    )


def _preserved_interpreter(model_path: Path) -> Any:
    Interpreter, OpResolverType = _litert()
    interpreter = Interpreter(
        model_path=str(model_path),
        experimental_op_resolver_type=(
            OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
        experimental_preserve_all_tensors=True,
    )
    interpreter.allocate_tensors()
    return interpreter


def _dequantized_tensor(
    interpreter: Any, tensor_index: int
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    details = {
        int(detail["index"]): detail for detail in interpreter.get_tensor_details()
    }
    detail = details[tensor_index]
    raw = interpreter.get_tensor(tensor_index).copy()
    parameters = detail["quantization_parameters"]
    scales = np.asarray(parameters["scales"], dtype=np.float32)
    zero_points = np.asarray(parameters["zero_points"], dtype=np.float32)
    if not len(scales):
        return (
            raw.astype(np.float32),
            raw,
            {
                "scale_count": 0,
                "quantized_dimension": 0,
            },
        )
    dimension = int(parameters["quantized_dimension"])
    if len(zero_points) == 1 and len(scales) > 1:
        zero_points = np.full(scales.shape, zero_points[0], dtype=np.float32)
    _require(
        len(zero_points) == len(scales),
        "tensor quantization scale/zero-point count changed",
    )
    shape = [1] * raw.ndim
    shape[dimension] = len(scales)
    value = (raw.astype(np.float32) - zero_points.reshape(shape)) * scales.reshape(
        shape
    )
    return (
        value,
        raw,
        {
            "scale_count": len(scales),
            "scale_minimum": float(np.min(scales)),
            "scale_maximum": float(np.max(scales)),
            "zero_point_minimum": int(np.min(zero_points)),
            "zero_point_maximum": int(np.max(zero_points)),
            "quantized_dimension": dimension,
        },
    )


def _stage_error_record(
    actual: np.ndarray,
    expected: np.ndarray,
    raw: np.ndarray,
    quantization: dict[str, object],
) -> dict[str, object]:
    _require(actual.shape == expected.shape, "recurrent trace tensor shape changed")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    expected_norm = float(np.linalg.norm(expected.astype(np.float64).reshape(-1)))
    record: dict[str, object] = {
        "shape": list(actual.shape),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "root_mean_square_error": float(np.sqrt(np.mean(difference * difference))),
        "relative_l2_error": float(
            np.linalg.norm(difference.reshape(-1)) / max(expected_norm, 1e-12)
        ),
        "float_minimum": float(np.min(expected)),
        "float_maximum": float(np.max(expected)),
        "dequantized_minimum": float(np.min(actual)),
        "dequantized_maximum": float(np.max(actual)),
        "quantization": quantization,
    }
    if np.issubdtype(raw.dtype, np.integer):
        bounds = np.iinfo(raw.dtype)
        record["quantized_endpoint_count"] = int(
            np.count_nonzero((raw == bounds.min) | (raw == bounds.max))
        )
        record["element_count"] = int(raw.size)
    return record


def _trace_baseline_window(
    interpreter: Any,
    state: dict[str, np.ndarray],
    window: np.ndarray,
    material_error: float,
) -> dict[str, object]:
    """Match the actual 302-op lowered graph to the Float recurrent stages."""
    operations = interpreter._get_ops_details()
    expected_prefix = ["FULLY_CONNECTED", "SPLIT"] + ["STRIDED_SLICE"] * 60
    _require(
        len(operations) == 302
        and [row["op_name"] for row in operations[:62]] == expected_prefix,
        "formal baseline graph cannot be mapped to the reviewed recurrent trace",
    )
    input_detail = interpreter.get_input_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    quantized_input = np.clip(
        np.rint(window / input_scale + input_zero), -128, 127
    ).astype(np.int8)[None, ...]
    interpreter.set_tensor(int(input_detail["index"]), quantized_input)
    interpreter.invoke()
    dequantized_input = (quantized_input.astype(np.float32) - input_zero) * input_scale
    float_trace, float_logits, float_probabilities = _float_recurrent_trace(
        state, window
    )
    input_projection, raw_projection, projection_quantization = _dequantized_tensor(
        interpreter, int(operations[0]["outputs"][0])
    )
    stage_order = (
        "hidden_projection",
        "reset_preactivation",
        "reset_gate",
        "reset_hidden_new",
        "candidate_preactivation",
        "candidate_gate",
        "hidden_difference",
        "update_preactivation",
        "update_gate",
        "update_delta",
        "hidden_state",
    )
    timesteps: list[dict[str, object]] = []
    first_recurrent_material: dict[str, object] | None = None
    first_hidden_material: int | None = None
    for timestep in range(20):
        stages: dict[str, object] = {
            "input": _stage_error_record(
                dequantized_input[0, timestep],
                window[timestep],
                quantized_input[0, timestep],
                {
                    "scale_count": 1,
                    "scale_minimum": float(input_scale),
                    "scale_maximum": float(input_scale),
                    "zero_point_minimum": int(input_zero),
                    "zero_point_maximum": int(input_zero),
                    "quantized_dimension": 0,
                },
            ),
            "input_projection": _stage_error_record(
                input_projection[timestep : timestep + 1],
                float_trace[timestep]["input_projection"],
                raw_projection[timestep : timestep + 1],
                projection_quantization,
            ),
        }
        if timestep == 0:
            base = 62
            offsets = {
                "reset_preactivation": 0,
                "reset_gate": 1,
                "reset_hidden_new": 2,
                "candidate_preactivation": 3,
                "candidate_gate": 4,
                "hidden_difference": 5,
                "update_preactivation": 6,
                "update_gate": 7,
                "update_delta": 8,
                "hidden_state": 9,
            }
        else:
            base = 72 + (timestep - 1) * 12
            offsets = {
                "hidden_projection": 0,
                "reset_preactivation": 2,
                "reset_gate": 3,
                "reset_hidden_new": 4,
                "candidate_preactivation": 5,
                "candidate_gate": 6,
                "hidden_difference": 7,
                "update_preactivation": 8,
                "update_gate": 9,
                "update_delta": 10,
                "hidden_state": 11,
            }
        for name, offset in offsets.items():
            operator = operations[base + offset]
            actual, raw, quantization = _dequantized_tensor(
                interpreter, int(operator["outputs"][0])
            )
            stages[name] = _stage_error_record(
                actual, float_trace[timestep][name], raw, quantization
            )
        if first_recurrent_material is None:
            for name in stage_order:
                if (
                    name in stages
                    and float(stages[name]["maximum_absolute_error"]) >= material_error
                ):
                    first_recurrent_material = {
                        "timestep": timestep,
                        "stage": name,
                        "maximum_absolute_error": stages[name][
                            "maximum_absolute_error"
                        ],
                    }
                    break
        if (
            first_hidden_material is None
            and float(stages["hidden_state"]["maximum_absolute_error"])
            >= material_error
        ):
            first_hidden_material = timestep
        timesteps.append({"timestep": timestep, "stages": stages})

    classifier, raw_classifier, classifier_quantization = _dequantized_tensor(
        interpreter, int(operations[300]["outputs"][0])
    )
    probability, raw_probability, probability_quantization = _dequantized_tensor(
        interpreter, int(operations[301]["outputs"][0])
    )
    return {
        "material_absolute_error": material_error,
        "first_recurrent_material_error": first_recurrent_material,
        "first_hidden_state_material_timestep": first_hidden_material,
        "input_projection_is_material_before_recurrence": any(
            float(row["stages"]["input_projection"]["maximum_absolute_error"])
            >= material_error
            for row in timesteps
        ),
        "timesteps": timesteps,
        "classifier_logits": _stage_error_record(
            classifier,
            float_logits,
            raw_classifier,
            classifier_quantization,
        ),
        "softmax_probabilities": _stage_error_record(
            probability,
            float_probabilities,
            raw_probability,
            probability_quantization,
        ),
    }


def _recurrent_sensitivity(
    state: dict[str, np.ndarray], window: np.ndarray
) -> dict[str, object]:
    """Measure local hidden-transition gain along one frozen Float trajectory."""
    values = torch.from_numpy(window.astype(np.float32, copy=False))
    weight_ih = torch.from_numpy(state["gru.weight_ih_l0"])
    weight_hh = torch.from_numpy(state["gru.weight_hh_l0"])
    bias_ih = torch.from_numpy(state["gru.bias_ih_l0"])
    bias_hh = torch.from_numpy(state["gru.bias_hh_l0"])
    input_reset, input_update, input_new = torch.nn.functional.linear(
        values, weight_ih, bias_ih
    ).chunk(3, dim=1)
    hidden = torch.zeros(32, dtype=torch.float32)
    jacobians = []
    for timestep in range(20):
        gate_inputs = (
            input_reset[timestep],
            input_update[timestep],
            input_new[timestep],
        )

        def transition(previous: torch.Tensor) -> torch.Tensor:
            hidden_reset, hidden_update, hidden_new = torch.nn.functional.linear(
                previous, weight_hh, bias_hh
            ).chunk(3, dim=0)
            reset = torch.sigmoid(gate_inputs[0] + hidden_reset)
            update = torch.sigmoid(gate_inputs[1] + hidden_update)
            new = torch.tanh(gate_inputs[2] + reset * hidden_new)
            return new + update * (previous - new)

        jacobians.append(torch.autograd.functional.jacobian(transition, hidden))
        hidden = transition(hidden).detach()
    local_norms = [
        float(torch.linalg.matrix_norm(jacobian, ord=2)) for jacobian in jacobians
    ]
    product = torch.eye(32, dtype=torch.float32)
    suffix_norms: list[float] = []
    for timestep in range(19, -1, -1):
        product = product @ jacobians[timestep]
        suffix_norms.append(float(torch.linalg.matrix_norm(product, ord=2)))
    suffix_norms.reverse()
    return {
        "local_hidden_jacobian_spectral_norm_by_timestep": local_norms,
        "maximum_local_hidden_jacobian_spectral_norm": max(local_norms),
        "hidden_jacobian_product_norm_from_timestep_by_timestep": suffix_norms,
        "full_20_step_hidden_jacobian_product_norm": suffix_norms[0],
        "maximum_suffix_hidden_jacobian_product_norm": max(suffix_norms),
        "classifier_weight_spectral_norm": float(
            torch.linalg.matrix_norm(
                torch.from_numpy(state["classifier.weight"]), ord=2
            )
        ),
    }


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
        "original_weights_kib": _match_float(
            r"Original Weights Size\s+([0-9.]+) KiB", text
        ),
        "npu_encoded_weights_kib": _match_float(
            r"NPU Encoded Weights Size\s+([0-9.]+) KiB", text
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
    projection_block_width: int | None = None,
) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    dequantized_outputs: list[np.ndarray] = []
    suffix = "probability" if include_softmax else "logits"
    for seed, state in zip(seeds, states):
        payload = _int8_tflite(
            state,
            calibration,
            include_softmax=include_softmax,
            projection_block_width=projection_block_width,
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
                    state,
                    calibration,
                    include_softmax=include_softmax,
                    projection_block_width=projection_block_width,
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
        distribution["int8_at_maximum_error"] = float(probabilities[index, error_index])
        probability_by_seed[str(seed)] = distribution
    parity: dict[str, object] = {
        "member_hazard_probability": _error_distribution(
            probabilities, golden["member_hazard_probability"]
        ),
        "member_hazard_probability_by_seed": probability_by_seed,
        "ensemble_hazard_probability": _error_distribution(
            ensemble, golden["ensemble_hazard_probability"]
        ),
        "threshold_crossing": _exact_parity(crossing, golden["threshold_crossing"]),
        "consecutive_threshold_count": _exact_parity(
            counts, golden["consecutive_threshold_count"]
        ),
        "reflex_required": _exact_parity(reflex, golden["reflex_required"]),
        "reflex_onset": _exact_parity(onset, golden["reflex_onset"]),
    }
    if logits is not None:
        parity["member_logits"] = _error_distribution(logits, golden["member_logits"])
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
        alternative_directory = _resolve_output(root, settings["alternative_directory"])
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
    above_index = int(
        np.flatnonzero(above)[np.argmin(expected_ensemble[above] - threshold)]
    )
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
        "onset_mismatch_count": selected["parity"]["reflex_onset"]["mismatch_count"],
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
    status = "INT8_QUANTIZATION_AND_PARITY_PASS" if numerical_pass else M3_VERDICT

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


def _contract_assessment(
    evaluation: dict[str, object], contract_settings: dict[str, object]
) -> dict[str, object]:
    parity = evaluation["parity"]
    member_records = parity["member_hazard_probability_by_seed"]
    ensemble_record = parity["ensemble_hazard_probability"]
    input_record = evaluation["members"][0]["io"]["input_quantization"]
    checks: dict[str, dict[str, object]] = {
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
    for record in checks.values():
        observed = float(
            record.get("observed", record.get("observed_maximum_across_members"))
        )
        record["status"] = "PASS" if observed <= float(record["maximum"]) else "FAIL"
    exact_names = (
        "threshold_crossing",
        "consecutive_threshold_count",
        "reflex_required",
        "reflex_onset",
    )
    continuous_status = (
        "PASS" if all(row["status"] == "PASS" for row in checks.values()) else "FAIL"
    )
    discrete_status = (
        "PASS"
        if all(parity[name]["status"] == "PASS" for name in exact_names)
        else "FAIL"
    )
    return {
        "status": (
            "PASS"
            if continuous_status == "PASS" and discrete_status == "PASS"
            else "FAIL"
        ),
        "continuous_status": continuous_status,
        "discrete_status": discrete_status,
        "checks": checks,
        "contract_changed_for_recovery": False,
    }


def _training_diagnostic(
    evaluation: dict[str, object],
    representative: np.ndarray,
    expected_probabilities: np.ndarray,
) -> dict[str, object]:
    actual_probabilities = []
    saturation = []
    for member in evaluation["members"]:
        values, io = _run_int8_model(Path(member["path"]), representative)
        actual_probabilities.append(values[:, 1].astype(np.float64))
        saturation.append(io["input_quantization"])
    actual = np.stack(actual_probabilities)
    expected_ensemble = np.mean(expected_probabilities, axis=0, dtype=np.float64)
    actual_ensemble = np.mean(actual, axis=0, dtype=np.float64)
    return {
        "source": "all_2597_frozen_TRAIN_derived_calibration_windows",
        "window_count": int(len(representative)),
        "canonical_golden_used": False,
        "member_hazard_probability": _error_distribution(
            actual, expected_probabilities
        ),
        "member_hazard_probability_by_seed": {
            str(seed): _error_distribution(actual[index], expected_probabilities[index])
            for index, seed in enumerate(MEMBER_SEEDS)
        },
        "ensemble_hazard_probability": _error_distribution(
            actual_ensemble, expected_ensemble
        ),
        "input_quantization": saturation[0],
    }


def _masked_error_distribution(
    actual: np.ndarray, expected: np.ndarray, mask: np.ndarray
) -> dict[str, object]:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {"count": 0, "available": False}
    return {
        "count": count,
        "available": True,
        **_error_distribution(actual[mask], expected[mask]),
    }


def _regime_distributions(
    actual: np.ndarray, expected: np.ndarray, threshold: float
) -> dict[str, object]:
    regimes = {
        "benign_or_normal_probability_at_most_0_10": expected <= 0.10,
        "transition_probability_between_0_10_and_threshold": (
            (expected > 0.10) & (expected < threshold)
        ),
        "threshold_vicinity_within_0_02": np.abs(expected - threshold) <= 0.02,
        "strong_hazard_at_or_above_threshold": expected >= threshold,
    }
    return {
        name: _masked_error_distribution(actual, expected, mask)
        for name, mask in regimes.items()
    }


def _hybrid_quantization_ablation(
    model_path: Path,
    state: dict[str, np.ndarray],
    windows: np.ndarray,
    expected: np.ndarray,
) -> dict[str, object]:
    """Separate IO/weight error from recurrent activation quantization."""
    interpreter = _interpreter(model_path)
    operations = interpreter._get_ops_details()
    _require(
        len(operations) == 302
        and operations[0]["op_name"] == "FULLY_CONNECTED"
        and operations[72]["op_name"] == "FULLY_CONNECTED"
        and operations[300]["op_name"] == "FULLY_CONNECTED",
        "formal baseline graph changed before hybrid ablation",
    )
    input_detail = interpreter.get_input_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    quantized = np.clip(np.rint(windows / input_scale + input_zero), -128, 127).astype(
        np.int8
    )
    dequantized_input = (quantized.astype(np.float32) - input_zero) * input_scale
    quantized_state = {name: value.copy() for name, value in state.items()}
    for name, operation_index in (
        ("gru.weight_ih_l0", 0),
        ("gru.weight_hh_l0", 72),
        ("classifier.weight", 300),
    ):
        weight_index = int(operations[operation_index]["inputs"][1])
        value, _, _ = _dequantized_tensor(interpreter, weight_index)
        _require(value.shape == state[name].shape, "quantized weight shape changed")
        quantized_state[name] = value.astype(np.float32, copy=False)
    scenarios = {
        "int8_io_only_float_weights_and_intermediates": (
            dequantized_input,
            state,
        ),
        "int8_per_axis_weights_only_float_io_and_intermediates": (
            windows,
            quantized_state,
        ),
        "int8_io_and_weights_float_recurrent_intermediates": (
            dequantized_input,
            quantized_state,
        ),
    }
    return {
        name: _error_distribution(
            _float_member_probabilities(selected_state, selected_windows), expected
        )
        for name, (selected_windows, selected_state) in scenarios.items()
    }


def _partitioned_hidden_trace(
    model_path: Path,
    state: dict[str, np.ndarray],
    window: np.ndarray,
    block_width: int,
    material_error: float,
) -> dict[str, object]:
    interpreter = _preserved_interpreter(model_path)
    input_detail = interpreter.get_input_details()[0]
    scale, zero = input_detail["quantization"]
    quantized = np.clip(np.rint(window / scale + zero), -128, 127).astype(np.int8)
    interpreter.set_tensor(int(input_detail["index"]), quantized[None, ...])
    interpreter.invoke()
    operations = interpreter._get_ops_details()
    details = {
        int(detail["index"]): detail for detail in interpreter.get_tensor_details()
    }
    blocks_per_gate = 32 // block_width
    block_count = 3 * blocks_per_gate
    input_projection_ops = [
        operation
        for operation in operations
        if operation["op_name"] == "FULLY_CONNECTED"
        and list(details[int(operation["outputs"][0])]["shape"]) == [20, block_width]
    ]
    hidden_concat_ops = [
        operation
        for operation in operations
        if operation["op_name"] == "CONCATENATION"
        and list(details[int(operation["outputs"][0])]["shape"]) == [1, 32]
    ]
    _require(
        len(input_projection_ops) == block_count and len(hidden_concat_ops) == 20,
        "partitioned recovery graph trace changed",
    )
    float_trace, float_logits, float_probability = _float_recurrent_trace(state, window)
    float_projection = np.concatenate(
        [row["input_projection"] for row in float_trace], axis=0
    )
    projection_blocks = np.split(float_projection, block_count, axis=1)
    projection_error = []
    for block, (operation, expected) in enumerate(
        zip(input_projection_ops, projection_blocks)
    ):
        actual, raw, quantization = _dequantized_tensor(
            interpreter, int(operation["outputs"][0])
        )
        projection_error.append(
            {
                "block": block,
                **_stage_error_record(actual, expected, raw, quantization),
            }
        )
    hidden_error = []
    for timestep, operation in enumerate(hidden_concat_ops):
        actual, raw, quantization = _dequantized_tensor(
            interpreter, int(operation["outputs"][0])
        )
        hidden_error.append(
            {
                "timestep": timestep,
                **_stage_error_record(
                    actual,
                    float_trace[timestep]["hidden_state"],
                    raw,
                    quantization,
                ),
            }
        )
    classifier_ops = [
        operation
        for operation in operations
        if operation["op_name"] == "FULLY_CONNECTED"
        and list(details[int(operation["outputs"][0])]["shape"]) == [1, 2]
    ]
    softmax_ops = [
        operation for operation in operations if operation["op_name"] == "SOFTMAX"
    ]
    _require(
        len(classifier_ops) == len(softmax_ops) == 1,
        "partitioned recovery output graph changed",
    )
    classifier, raw_classifier, classifier_q = _dequantized_tensor(
        interpreter, int(classifier_ops[0]["outputs"][0])
    )
    probability, raw_probability, probability_q = _dequantized_tensor(
        interpreter, int(softmax_ops[0]["outputs"][0])
    )
    return {
        "projection_block_width": block_width,
        "input_projection_blocks": projection_error,
        "hidden_state_by_timestep": hidden_error,
        "first_hidden_state_material_timestep": next(
            (
                int(row["timestep"])
                for row in hidden_error
                if float(row["maximum_absolute_error"]) >= material_error
            ),
            None,
        ),
        "classifier_logits": _stage_error_record(
            classifier, float_logits, raw_classifier, classifier_q
        ),
        "softmax_probabilities": _stage_error_record(
            probability, float_probability, raw_probability, probability_q
        ),
    }


def evaluate_int8_recovery(
    repository_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path | None = None,
) -> dict[str, object]:
    """Localize formal INT8 recurrent error and assess focused PTQ recovery."""
    root = repository_root.resolve()
    bundle, config, model_manifest, _ = validate_reference_bundle(root, config_path)
    formal_settings = config.get("formal_int8")
    settings = config.get("int8_recovery")
    _require(
        isinstance(formal_settings, dict) and isinstance(settings, dict),
        "M3.1 recovery config is missing",
    )
    seeds = [int(value) for value in formal_settings["member_seeds"]]
    _require(seeds == list(MEMBER_SEEDS), "M3.1 member order changed")
    with np.load(
        bundle / "calibration_inputs/int8_representative.npz", allow_pickle=False
    ) as payload:
        representative = payload["model_windows"].copy()
    percentile = float(
        formal_settings["calibration_policy"]["symmetric_absolute_percentile"]
    )
    _require(percentile == 99.0, "formal calibration policy changed before M3.1")
    clipping_bound = float(np.percentile(np.abs(representative), percentile))
    calibration = np.clip(representative, -clipping_bound, clipping_bound).astype(
        np.float32, copy=False
    )
    with np.load(
        bundle / "golden_outputs/deployment_runtime_chain.npz", allow_pickle=False
    ) as payload:
        golden = {name: payload[name].copy() for name in payload.files}
    windows = golden["model_windows"]
    models = load_ensemble(bundle, model_manifest)
    states = [_state_arrays(model) for model in models]
    training_float = np.stack(
        [_float_member_probabilities(state, representative) for state in states]
    )

    if output_root is None:
        model_directory = _resolve_output(root, settings["model_directory"])
        vela_directory = _resolve_output(root, settings["vela_directory"])
        result_path = _resolve_output(root, settings["result_path"])
    else:
        selected_root = output_root.resolve()
        model_directory = selected_root / "quantized/recovery"
        vela_directory = selected_root / "vela/recovery"
        result_path = selected_root / "int8_recurrent_error_localization.json"

    baseline = _evaluate_int8_ensemble(
        states=states,
        seeds=seeds,
        calibration=calibration,
        windows=windows,
        golden=golden,
        output_directory=model_directory / "formal_baseline_reproduction",
        include_softmax=True,
        expected_operators=formal_settings["expected_probability_operators"],
        repeat_conversion=True,
    )
    expected_hashes = settings["baseline_artifact_sha256"]
    for member in baseline["members"]:
        seed = str(member["seed"])
        _require(
            member["sha256"] == expected_hashes[seed],
            f"M3 baseline artifact was not reproduced for seed {seed}",
        )
    baseline_contract = _contract_assessment(
        baseline, formal_settings["numerical_contract"]
    )
    _require(
        baseline_contract["status"] == "FAIL"
        and baseline_contract["discrete_status"] == "PASS",
        "M3 failure was not reproduced",
    )

    candidate_runtime: dict[str, dict[str, object]] = {}
    candidate_evidence: dict[str, dict[str, object]] = {}
    for candidate in settings["projection_partition_candidates"]:
        name = str(candidate["name"])
        width = int(candidate["block_width"])
        evaluation = _evaluate_int8_ensemble(
            states=states,
            seeds=seeds,
            calibration=calibration,
            windows=windows,
            golden=golden,
            output_directory=model_directory / name,
            include_softmax=True,
            expected_operators=candidate["expected_operators"],
            repeat_conversion=False,
            projection_block_width=width,
        )
        training = _training_diagnostic(evaluation, representative, training_float)
        candidate_runtime[name] = {
            "evaluation": evaluation,
            "block_width": width,
        }
        candidate_evidence[name] = {
            "projection_block_width": width,
            "mathematical_change": "none_projection_rows_only_partitioned",
            "weights_changed": False,
            "calibration_changed": False,
            "training_diagnostic": training,
            "golden_evaluation": {
                key: value for key, value in evaluation.items() if key != "actual"
            },
            "contract": _contract_assessment(
                evaluation, formal_settings["numerical_contract"]
            ),
        }

    selected_name = min(
        candidate_evidence,
        key=lambda name: (
            float(
                candidate_evidence[name]["training_diagnostic"][
                    "ensemble_hazard_probability"
                ]["p95_absolute_error"]
            ),
            int(candidate_evidence[name]["projection_block_width"]),
        ),
    )
    selected_runtime = candidate_runtime[selected_name]
    selected_evaluation = selected_runtime["evaluation"]
    selected_contract = candidate_evidence[selected_name]["contract"]
    for state, member in zip(states, selected_evaluation["members"]):
        repeated = _int8_tflite(
            state,
            calibration,
            include_softmax=True,
            projection_block_width=int(selected_runtime["block_width"]),
        )
        member["repeated_conversion_byte_identical"] = (
            repeated == Path(member["path"]).read_bytes()
        )
        _require(
            member["repeated_conversion_byte_identical"],
            f"M3.1 selected conversion is not deterministic: {member['seed']}",
        )

    expected_ensemble = golden["ensemble_hazard_probability"]
    baseline_actual = baseline["actual"]
    selected_actual = selected_evaluation["actual"]
    threshold = float(formal_settings["threshold"])
    regime_comparison = {
        "baseline": _regime_distributions(
            baseline_actual["ensemble"], expected_ensemble, threshold
        ),
        "selected_partial_recovery": _regime_distributions(
            selected_actual["ensemble"], expected_ensemble, threshold
        ),
    }

    material_error = float(settings["material_absolute_error"])
    baseline_trace = []
    worst_indices = []
    for index, seed in enumerate(seeds):
        member_record = baseline["parity"]["member_hazard_probability_by_seed"][
            str(seed)
        ]
        window_index = int(member_record["maximum_error_index"])
        worst_indices.append(window_index)
        interpreter = _preserved_interpreter(Path(baseline["members"][index]["path"]))
        baseline_trace.append(
            {
                "seed": seed,
                "window_index": window_index,
                "selection": "formal_baseline_member_maximum_probability_error",
                "float_hazard_probability": float(
                    golden["member_hazard_probability"][index, window_index]
                ),
                "int8_hazard_probability": float(
                    baseline_actual["probabilities"][index, window_index]
                ),
                "trace": _trace_baseline_window(
                    interpreter,
                    states[index],
                    windows[window_index],
                    material_error,
                ),
                "sensitivity": _recurrent_sensitivity(
                    states[index], windows[window_index]
                ),
            }
        )
    above = expected_ensemble >= threshold
    representative_indices = {
        "minimum_ensemble_probability": int(np.argmin(expected_ensemble)),
        "closest_below_threshold": int(
            np.flatnonzero(~above)[np.argmin(threshold - expected_ensemble[~above])]
        ),
        "maximum_ensemble_probability": int(np.argmax(expected_ensemble)),
    }
    diagnostic_member = 1
    diagnostic_interpreter = _preserved_interpreter(
        Path(baseline["members"][diagnostic_member]["path"])
    )
    for label, window_index in representative_indices.items():
        baseline_trace.append(
            {
                "seed": seeds[diagnostic_member],
                "window_index": window_index,
                "selection": label,
                "float_hazard_probability": float(
                    golden["member_hazard_probability"][diagnostic_member, window_index]
                ),
                "int8_hazard_probability": float(
                    baseline_actual["probabilities"][diagnostic_member, window_index]
                ),
                "trace": _trace_baseline_window(
                    diagnostic_interpreter,
                    states[diagnostic_member],
                    windows[window_index],
                    material_error,
                ),
            }
        )

    hybrid_ablation = {
        str(seed): _hybrid_quantization_ablation(
            Path(baseline["members"][index]["path"]),
            states[index],
            windows,
            golden["member_hazard_probability"][index],
        )
        for index, seed in enumerate(seeds)
    }
    selected_residual_trace = []
    for index, seed in enumerate(seeds):
        member_record = selected_evaluation["parity"][
            "member_hazard_probability_by_seed"
        ][str(seed)]
        window_index = int(member_record["maximum_error_index"])
        selected_residual_trace.append(
            {
                "seed": seed,
                "window_index": window_index,
                "float_hazard_probability": float(
                    golden["member_hazard_probability"][index, window_index]
                ),
                "int8_hazard_probability": float(
                    selected_actual["probabilities"][index, window_index]
                ),
                "trace": _partitioned_hidden_trace(
                    Path(selected_evaluation["members"][index]["path"]),
                    states[index],
                    windows[window_index],
                    int(selected_runtime["block_width"]),
                    material_error,
                ),
            }
        )

    vela = shutil.which("vela")
    if vela is None:
        raise RuntimeError("Vela executable is not installed or not on PATH")
    target = formal_settings["target"]
    mappings: dict[str, dict[str, object]] = {}
    for memory_mode in ("Shared_Sram", "Sram_Only"):
        rows: dict[str, object] = {}
        for member in selected_evaluation["members"]:
            seed = str(member["seed"])
            mapping = _run_vela(
                vela,
                Path(member["path"]),
                vela_directory / memory_mode.lower() / f"member_seed{seed}",
                str(target["accelerator_config"]),
                str(target["generic_system_config"]),
                memory_mode,
            )
            _require(
                mapping["cpu_operators"] == 0
                and mapping["npu_operators"]
                == int(settings["selected_expected_npu_operators"]),
                f"M3.1 selected candidate has U55 fallback: {seed}/{memory_mode}",
            )
            rows[seed] = mapping
        mappings[memory_mode] = rows
    vela_summary = {
        memory_mode: {
            "three_member_macs": sum(
                int(row["macs_per_inference"]) for row in rows.values()
            ),
            "three_member_npu_cycles": sum(
                int(row["npu_cycles"]) for row in rows.values()
            ),
            "three_member_total_cycles": sum(
                int(row["total_cycles"]) for row in rows.values()
            ),
            "maximum_member_sram_kib": max(
                float(row["sram_kib"]) for row in rows.values()
            ),
            "sum_compiled_model_bytes": sum(
                int(row["compiled_model"]["bytes"]) for row in rows.values()
            ),
        }
        for memory_mode, rows in mappings.items()
    }

    formal_rerun = selected_contract["status"] == "PASS"
    _require(not formal_rerun, "credible M3.1 recovery requires a formal M3 rerun")

    def without_actual(value: dict[str, object]) -> dict[str, object]:
        return {key: item for key, item in value.items() if key != "actual"}

    result: dict[str, object] = {
        "status": M31_VERDICT,
        "reference": {
            "candidate_id": model_manifest["candidate_id"],
            "engineering_role": model_manifest["engineering_role"],
            "scientific_verdict": model_manifest["scientific_status"]["verdict"],
            "research_release_commit": config["accepted_identity"][
                "research_release_commit"
            ],
            "calibration_sha256": config["accepted_identity"][
                "int8_calibration_sha256"
            ],
        },
        "calibration": {
            "source": "frozen exact effective TRAIN handoff",
            "window_count": int(len(representative)),
            "symmetric_absolute_percentile": percentile,
            "clipping_bound": clipping_bound,
            "policy_changed_from_formal_m3": False,
            "protected_holdout_access": False,
            "golden_used_for_candidate_selection": False,
        },
        "m3_baseline_reproduction": {
            "artifact_hashes_match": True,
            "contract": baseline_contract,
            "evaluation": without_actual(baseline),
        },
        "localization": {
            "actual_graph": (
                "302 static TFLite built-in operations; no high-level GRU operator"
            ),
            "material_absolute_error": material_error,
            "worst_window_indices_by_seed": dict(zip(map(str, seeds), worst_indices)),
            "representative_window_indices": representative_indices,
            "baseline_traces": baseline_trace,
            "hybrid_io_weight_ablation": hybrid_ablation,
        },
        "ptq_candidates": candidate_evidence,
        "selection": {
            "name": selected_name,
            "projection_block_width": selected_runtime["block_width"],
            "criterion": (
                "minimum ensemble p95 absolute error on all 2597 TRAIN-derived "
                "diagnostic windows; golden excluded"
            ),
            "golden_used": False,
            "calibration_policy_changed": False,
            "weights_or_semantics_changed": False,
            "byte_deterministic_all_members": all(
                bool(member["repeated_conversion_byte_identical"])
                for member in selected_evaluation["members"]
            ),
            "contract": selected_contract,
            "residual_traces": selected_residual_trace,
        },
        "golden_regime_comparison": regime_comparison,
        "vela": {
            "tool_version": _package_version("ethos-u-vela"),
            "accelerator_config": target["accelerator_config"],
            "system_config": target["generic_system_config"],
            "members": mappings,
            "three_member_summary": vela_summary,
            "timing_is_board_measured": False,
        },
        "formal_m3_rerun": {
            "performed": formal_rerun,
            "reason": (
                "not performed because the independently selected best PTQ "
                "candidate still fails the unchanged M3 numerical contract"
            ),
        },
        "root_cause_assessment": {
            "classification": (
                "per_tensor_projection_and_recurrent_activation_quantization_"
                "amplified_by_high_gain_hidden_dynamics"
            ),
            "input_saturation_primary_cause": False,
            "classifier_or_softmax_primary_cause": False,
            "ptq_only_full_int8_recovery_demonstrated": False,
            "research_intervention_required": True,
            "recommended_research_scope": (
                "reviewed QAT or a deployment-aware recurrent model change"
            ),
        },
        "boundary": {
            "m31_completed": True,
            "existing_m3_numerical_contract_passed": False,
            "m4_authorized": False,
            "firmware_started": False,
            "board_state_modified": False,
            "research_repository_modified": False,
            "protected_holdout_access": False,
        },
        "next_milestone": "RESEARCH_QUANTIZATION_AWARE_INTERVENTION_REVIEW",
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["result_path"] = str(result_path)
    return result


def freeze_non_release_hil_prototype(
    repository_root: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Freeze the reproduced M3.1 selection without changing formal status."""
    root = repository_root.resolve()
    _, config, model_manifest, _ = validate_reference_bundle(root, config_path)
    settings = config.get("non_release_hil_prototype")
    recovery_settings = config.get("int8_recovery")
    formal_settings = config.get("formal_int8")
    _require(
        isinstance(settings, dict)
        and isinstance(recovery_settings, dict)
        and isinstance(formal_settings, dict),
        "non-release prototype config is missing",
    )
    _require(settings.get("role") == PROTOTYPE_ROLE, "prototype role changed")
    flags = settings.get("flags")
    _require(
        isinstance(flags, dict)
        and flags
        == {
            "formal_m3_pass": False,
            "numerical_contract_pass": False,
            "scientific_release": False,
            "real_robot_supported": False,
            "production_ready": False,
            "safety_certified": False,
            "hil_path_only": True,
        },
        "prototype safety boundary changed",
    )

    recovery_path = _resolve_output(root, str(settings["recovery_result_path"]))
    _require(recovery_path.is_file(), "run evaluate-int8-recovery before freeze")
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    _require(recovery.get("status") == M31_VERDICT, "M3.1 verdict changed")
    _require(
        recovery.get("boundary", {}).get("existing_m3_numerical_contract_passed")
        is False
        and recovery.get("boundary", {}).get("m4_authorized") is False,
        "formal M3/M4 boundary changed",
    )
    selection = recovery.get("selection", {})
    _require(
        selection.get("name") == settings["selected_representation"]
        and selection.get("projection_block_width")
        == settings["projection_block_width"]
        and selection.get("byte_deterministic_all_members") is True
        and selection.get("golden_used") is False
        and selection.get("weights_or_semantics_changed") is False
        and selection.get("contract", {}).get("status") == "FAIL"
        and selection.get("contract", {}).get("discrete_status") == "PASS",
        "reproduced M3.1 selection does not match the prototype contract",
    )
    _require(
        recovery.get("calibration", {}).get("symmetric_absolute_percentile") == 99.0
        and recovery.get("calibration", {}).get("clipping_bound") == 4.132843623161313
        and recovery.get("calibration", {}).get("policy_changed_from_formal_m3")
        is False,
        "prototype calibration policy changed",
    )

    candidate = recovery.get("ptq_candidates", {}).get(
        settings["selected_representation"], {}
    )
    golden_evaluation = candidate.get("golden_evaluation", {})
    members = golden_evaluation.get("members", [])
    seeds = [int(value) for value in settings["runtime"]["ensemble_order"]]
    _require(seeds == list(MEMBER_SEEDS), "prototype ensemble order changed")
    _require(
        [int(member["seed"]) for member in members] == seeds,
        "recovery evidence member order changed",
    )
    expected_hashes = settings["source_artifact_sha256"]
    expected_graph = settings["graph"]
    expected_runtime = settings["runtime"]
    source_directory = _resolve_output(root, str(settings["source_directory"]))
    prototype_directory = _resolve_output(root, str(settings["directory"]))
    model_directory = prototype_directory / "models"
    model_directory.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, object]] = []
    for seed, evidence in zip(seeds, members):
        filename = f"member_seed{seed}_int8_probability.tflite"
        source = source_directory / filename
        expected_hash = str(expected_hashes[str(seed)])
        _require(
            source.is_file() and sha256_file(source) == expected_hash,
            f"selected M3.1 artifact identity changed: {seed}",
        )
        graph = inspect_tflite(source)
        _require(
            graph["operator_count"] == expected_graph["operator_count"]
            and graph["operators"] == expected_graph["operators"],
            f"prototype graph semantics changed: {seed}",
        )
        input_record = graph["inputs"][0]
        output_record = graph["outputs"][0]
        _require(
            input_record["shape"] == expected_runtime["input_shape"]
            and input_record["dtype"] == expected_runtime["input_dtype"]
            and input_record["quantization"]
            == {
                "scale": expected_runtime["input_scale"],
                "zero_point": expected_runtime["input_zero_point"],
            }
            and output_record["shape"] == expected_runtime["output_shape"]
            and output_record["dtype"] == expected_runtime["output_dtype"]
            and output_record["quantization"]
            == {
                "scale": expected_runtime["output_scale"],
                "zero_point": expected_runtime["output_zero_point"],
            },
            f"prototype TFLite IO contract changed: {seed}",
        )
        _require(
            evidence["sha256"] == expected_hash
            and evidence["repeated_conversion_byte_identical"] is True,
            f"M3.1 reproducibility evidence changed: {seed}",
        )
        destination = model_directory / filename
        shutil.copyfile(source, destination)
        _require(
            sha256_file(destination) == expected_hash,
            f"prototype freeze copy failed checksum verification: {seed}",
        )
        artifacts.append(
            {
                "seed": seed,
                "path": str(destination.relative_to(root)),
                "bytes": destination.stat().st_size,
                "sha256": expected_hash,
            }
        )

    member_parity = golden_evaluation["parity"]["member_hazard_probability_by_seed"]
    for seed in seeds:
        _require(
            member_parity[str(seed)]["maximum_absolute_error"]
            == settings["expected_member_maximum_absolute_error"][str(seed)],
            f"prototype member metric changed: {seed}",
        )
    ensemble_parity = golden_evaluation["parity"]["ensemble_hazard_probability"]
    for name, expected in settings["expected_ensemble"].items():
        _require(ensemble_parity[name] == expected, f"ensemble metric changed: {name}")
    exact_parity = {
        name: bool(golden_evaluation["parity"][name]["exact"])
        for name in (
            "threshold_crossing",
            "consecutive_threshold_count",
            "reflex_required",
            "reflex_onset",
        )
    }
    _require(all(exact_parity.values()), "prototype discrete parity changed")

    accepted = config["accepted_identity"]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "prototype_id": settings["prototype_id"],
        "role": PROTOTYPE_ROLE,
        **flags,
        "formal_status": M31_VERDICT,
        "m4_authorized": False,
        "scientific_verdict": accepted["scientific_verdict"],
        "simulation_status": accepted["simulation_status"],
        "reference": {
            "release_id": accepted["release_id"],
            "research_repository": accepted["research_repository"],
            "research_release_commit": accepted["research_release_commit"],
            "release_manifest_sha256": accepted["release_manifest_sha256"],
            "architecture_sha256": accepted["architecture_sha256"],
            "feature_schema_sha256": accepted["feature_schema_sha256"],
            "normalizer_sha256": accepted["normalizer_sha256"],
            "calibration_sha256": accepted["int8_calibration_sha256"],
            "checkpoint_sha256": accepted["checkpoint_sha256"],
        },
        "reproduction": {
            "m31_evidence_path": str(recovery_path.relative_to(root)),
            "m31_evidence_sha256": sha256_file(recovery_path),
            "selected_representation": settings["selected_representation"],
            "projection_block_width": settings["projection_block_width"],
            "calibration_source": "frozen exact effective TRAIN handoff",
            "calibration_window_count": 2597,
            "symmetric_absolute_percentile": 99.0,
            "clipping_bound": 4.132843623161313,
            "golden_used_for_selection": False,
            "byte_deterministic_all_members": True,
            "weights_or_semantics_changed": False,
        },
        "runtime_contract": expected_runtime,
        "graph": expected_graph,
        "artifacts": artifacts,
        "canonical_golden_parity": {
            "member_maximum_absolute_error": {
                str(seed): member_parity[str(seed)]["maximum_absolute_error"]
                for seed in seeds
            },
            "ensemble": settings["expected_ensemble"],
            "exact": exact_parity,
            "numerical_contract": "FAIL",
            "failure": "member_probability_maximum_absolute_error_gt_0.10",
        },
        "prohibited_source": "DEPLOYMENT_AWARE_QAT_TRAIN_ACCEPTANCE_FAIL",
    }
    manifest_path = prototype_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = root / "reports/int8_16ch_hil_prototype_freeze.json"
    report_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": "NON_RELEASE_HIL_PATH_PROTOTYPE_FROZEN",
        "prototype_manifest": str(manifest_path),
        "prototype_manifest_sha256": sha256_file(manifest_path),
        "artifact_sha256": {str(row["seed"]): row["sha256"] for row in artifacts},
        "formal_status": M31_VERDICT,
        "formal_m3_pass": False,
        "m4_authorized": False,
    }


def _ml_coretools_config(
    source: Path, key: str, target: dict[str, object]
) -> dict[str, object]:
    requirements = {
        "accuracy": {
            "float": 100.0,
            "int16x16": 100.0,
            "int16x8": 100.0,
            "int8x8": 100.0,
        },
        "cpu_cycles": {name: 0 for name in ("float", "int16x16", "int16x8", "int8x8")},
        "mae": {name: 1.0 for name in ("float", "int16x16", "int16x8", "int8x8")},
        "scratch_mem_kb": {
            name: 10.0 for name in ("float", "int16x16", "int16x8", "int8x8")
        },
        "scratch_mem_opt_kb": {
            name: 10.0 for name in ("float", "int16x16", "int16x8", "int8x8")
        },
    }
    random_data = {
        "data_mode": "random",
        "max_samples": 1,
        # CoreTools 3.0 reads the outer fields while its schema also describes
        # a nested data_config.  Keep both for an exact, schema-valid invocation.
        "data_config": {"data_mode": "random", "max_samples": 1},
    }
    return {
        "version": "2.0.0",
        "model": {
            "load_model": str(source),
            "key_model": key,
            "model_task": "classification",
            "is_model_rnn": False,
            "optimization_settings": {
                "target": "PSE84_M55_U55",
                "interpreter": "tflm",
                "quantization": ["int8x8"],
                "sparsity": False,
                "ethos_u_npu_vela_options": {
                    "system_config": target["system_config"],
                    "memory_mode": target["memory_mode"],
                    "arena_cache_size": target["arena_cache_size"],
                    "max_block_dependency": target["max_block_dependency"],
                    "optimization_strategy": target["optimization_strategy"],
                    "recursion_limit": target["recursion_limit"],
                    "tensor_allocator": target["tensor_allocator"],
                },
            },
        },
        "data": {
            "calibration_data": random_data,
            "validation_data": random_data,
        },
        "cdv_requirements": {
            "tflm": requirements,
            "ifx": requirements,
        },
    }


def _compiled_ethosu_buffers(model_path: Path) -> dict[str, int]:
    """Read exact command/constant buffer sizes from a Vela TFLite flatbuffer."""
    _tensorflow()
    from tensorflow.lite.python import schema_py_generated as schema

    payload = model_path.read_bytes()
    model = schema.Model.GetRootAsModel(payload, 0)
    _require(model.SubgraphsLength() == 1, "compiled model subgraph count changed")
    subgraph = model.Subgraphs(0)
    _require(subgraph.OperatorsLength() == 1, "compiled model is not one Ethos-U op")
    opcode = model.OperatorCodes(subgraph.Operators(0).OpcodeIndex())
    custom_code = opcode.CustomCode()
    _require(custom_code == b"ethos-u", "compiled model does not contain Ethos-U")
    sizes: dict[str, int] = {}
    for index in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(index)
        name_value = tensor.Name()
        name = "" if name_value is None else name_value.decode("utf-8")
        length = int(model.Buffers(tensor.Buffer()).DataLength())
        if name.endswith("_command_stream"):
            sizes["command_stream_bytes"] = length
        elif name.endswith("_flash"):
            sizes["constant_flash_tensor_bytes"] = length
    _require(
        sizes.get("command_stream_bytes", 0) > 0
        and sizes.get("constant_flash_tensor_bytes", 0) > 0,
        "compiled Ethos-U buffers are missing",
    )
    return sizes


def compile_non_release_hil_prototype_for_e84(
    repository_root: Path,
    ml_coretools: Path,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path | None = None,
) -> dict[str, object]:
    """Compile the frozen prototype with the official PSE84 ML Pack preset."""
    root = repository_root.resolve()
    _, config, _, _ = validate_reference_bundle(root, config_path)
    prototype_settings = config.get("non_release_hil_prototype")
    settings = config.get("target_vela")
    _require(
        isinstance(prototype_settings, dict) and isinstance(settings, dict),
        "target Vela configuration is missing",
    )
    executable = ml_coretools.expanduser().resolve()
    _require(executable.is_file(), f"ML CoreTools executable not found: {executable}")
    version_run = subprocess.run(
        [str(executable), "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _require(version_run.returncode == 0, "ML CoreTools version query failed")
    _require(
        f"ml-coretools {settings['ml_coretools_version']}" in version_run.stdout
        and f"ethos-u-vela {settings['vela_version']}" in version_run.stdout,
        "official ML CoreTools/Vela version does not match the pinned configuration",
    )

    prototype_directory = _resolve_output(root, str(prototype_settings["directory"]))
    manifest_path = prototype_directory / "manifest.json"
    _require(manifest_path.is_file(), "freeze the non-release prototype first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("role") == PROTOTYPE_ROLE
        and manifest.get("formal_m3_pass") is False
        and manifest.get("numerical_contract_pass") is False,
        "prototype/formal status separation changed",
    )
    target = settings["target"]
    _require(isinstance(target, dict), "target Vela settings are invalid")
    if output_root is None:
        output_directory = _resolve_output(root, str(settings["output_directory"]))
        report_path = _resolve_output(root, str(settings["report_path"]))
    else:
        output_directory = output_root.resolve()
        report_path = output_directory / "e84_target_vela_compilation.json"
    output_directory.mkdir(parents=True, exist_ok=True)

    expected = settings["expected"]
    members: list[dict[str, object]] = []
    for artifact in manifest["artifacts"]:
        seed = int(artifact["seed"])
        source = root / str(artifact["path"])
        _require(
            sha256_file(source) == artifact["sha256"],
            f"prototype source hash changed: {seed}",
        )
        member_directory = output_directory / f"member_seed{seed}"
        member_directory.mkdir(parents=True, exist_ok=True)
        key = f"fastreflex_seed{seed}"
        tool_config = _ml_coretools_config(source, key, target)
        tool_config_path = member_directory / "ml_coretools_config.json"
        tool_config_path.write_text(
            json.dumps(tool_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = [
            str(executable),
            "--verbose",
            "--config",
            str(tool_config_path),
            "--output",
            str(member_directory),
            "--mode",
            "deploy",
        ]
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        console_path = member_directory / "ml_coretools_console.log"
        console_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"ML CoreTools failed for seed {seed}; see {console_path}"
            )
        compiled_matches = list(
            (member_directory / "model_gen_dir").glob("*_vela_int8x8.tflite")
        )
        info_matches = list((member_directory / "info").glob("*_vela_int8x8.txt"))
        metrics_matches = list((member_directory / "info").glob("*_model_metrics.json"))
        _require(
            len(compiled_matches) == len(info_matches) == len(metrics_matches) == 1,
            f"unexpected ML CoreTools output set for seed {seed}",
        )
        compiled = compiled_matches[0]
        summary = _vela_summary(info_matches[0].read_text(encoding="utf-8"))
        metrics = json.loads(metrics_matches[0].read_text(encoding="utf-8"))
        _require(
            summary["cpu_operators"] == expected["cpu_operators_per_member"]
            and summary["npu_operators"] == expected["npu_operators_per_member"],
            f"target Vela CPU fallback/operator count changed: {seed}",
        )
        members.append(
            {
                "seed": seed,
                "source": {
                    "path": str(source.relative_to(root)),
                    "bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                },
                "command": command,
                "vela_options": {
                    "accelerator_config": target["accelerator_config"],
                    "system_config": target["system_config"],
                    "memory_mode": target["memory_mode"],
                    "arena_cache_size": target["arena_cache_size"],
                    "max_block_dependency": target["max_block_dependency"],
                    "optimise": target["optimization_strategy"],
                    "tensor_allocator": target["tensor_allocator"],
                },
                "compiled_model": {
                    "path": str(compiled.relative_to(root))
                    if root in compiled.parents
                    else str(compiled),
                    "bytes": compiled.stat().st_size,
                    "sha256": sha256_file(compiled),
                },
                "tensor_arena_bytes": int(metrics["arena_size"]["int8x8"]),
                **_compiled_ethosu_buffers(compiled),
                **summary,
            }
        )

    def integer_sum(name: str) -> int:
        return sum(int(member[name]) for member in members)

    def float_sum(name: str) -> float:
        return sum(float(member[name]) for member in members)

    result: dict[str, object] = {
        "status": "E84_TARGET_VELA_COMPILATION_PASS",
        "role": PROTOTYPE_ROLE,
        "formal_status": M31_VERDICT,
        "formal_m3_pass": False,
        "numerical_contract_pass": False,
        "toolchain": {
            "machine_learning_pack_version": settings["machine_learning_pack_version"],
            "machine_learning_pack_artifact_uuid": settings[
                "machine_learning_pack_artifact_uuid"
            ],
            "machine_learning_pack_sha256": settings["machine_learning_pack_sha256"],
            "ml_coretools_version": settings["ml_coretools_version"],
            "vela_version": settings["vela_version"],
            "version_output": version_run.stdout.strip(),
            "executable": str(executable),
            "vela_config_path": settings["vela_config_path"],
        },
        "official_example": settings["official_example"],
        "bsp": settings["bsp"],
        "target": target,
        "placement": {
            "vela_logical_constant_area": "OnChipFlash using SRAM characteristics",
            "firmware_model_section": target["firmware_model_section"],
            "application_execution": target["application_xip"],
            "qualification": (
                "Vela estimates are compiler estimates. Firmware linker placement "
                "and physical runtime memory are reported separately."
            ),
        },
        "members": members,
        "three_member_sequential_summary": {
            "compiled_model_bytes": sum(
                int(member["compiled_model"]["bytes"]) for member in members
            ),
            "command_stream_bytes": integer_sum("command_stream_bytes"),
            "constant_flash_tensor_bytes": integer_sum("constant_flash_tensor_bytes"),
            "original_weights_kib": round(float_sum("original_weights_kib"), 2),
            "npu_encoded_weights_kib": round(float_sum("npu_encoded_weights_kib"), 2),
            "macs": integer_sum("macs_per_inference"),
            "npu_cycles": integer_sum("npu_cycles"),
            "total_cycles": integer_sum("total_cycles"),
            "estimated_inference_ms": round(
                float_sum("tool_estimated_inference_ms"), 3
            ),
            "peak_sram_kib_if_arena_reused": max(
                float(member["sram_kib"]) for member in members
            ),
            "tensor_arena_bytes_if_reused": max(
                int(member["tensor_arena_bytes"]) for member in members
            ),
            "cpu_operators": integer_sum("cpu_operators"),
            "npu_operators": integer_sum("npu_operators"),
        },
        "boundary": {
            "compiler_estimate_only": True,
            "board_execution_measured": False,
            "scientific_verdict_changed": False,
            "formal_numerical_contract_changed": False,
            "m4_authorized": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["report_path"] = str(report_path)
    return result


def _c_float_array(name: str, values: np.ndarray) -> str:
    encoded = ",\n    ".join(float(np.float32(value)).hex() + "f" for value in values)
    return f"const float {name}[80] = {{\n    {encoded}\n}};\n"


def stage_e84_firmware_assets(
    repository_root: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Stage checksum-verified generated models and normalizer for the build."""
    root = repository_root.resolve()
    bundle, config, _, _ = validate_reference_bundle(root, config_path)
    prototype = config.get("non_release_hil_prototype")
    target = config.get("target_vela")
    firmware = config.get("firmware")
    _require(
        isinstance(prototype, dict)
        and isinstance(target, dict)
        and isinstance(firmware, dict),
        "firmware staging configuration is missing",
    )
    report_path = _resolve_output(root, str(target["report_path"]))
    _require(report_path.is_file(), "run compile-target-vela before staging firmware")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        report.get("status") == "E84_TARGET_VELA_COMPILATION_PASS"
        and report.get("role") == PROTOTYPE_ROLE
        and report.get("formal_m3_pass") is False
        and report.get("numerical_contract_pass") is False,
        "target Vela evidence or prototype boundary changed",
    )
    output = _resolve_output(root, str(firmware["generated_asset_directory"]))
    model_output = output / "mtb_ml_models"
    if output.exists():
        shutil.rmtree(output)
    model_output.mkdir(parents=True)

    staged: list[dict[str, object]] = []
    vela_root = _resolve_output(root, str(target["output_directory"]))
    for member in report["members"]:
        seed = int(member["seed"])
        prefix = f"FASTREFLEX_SEED{seed}_tflm_model_int8x8"
        source_directory = vela_root / f"member_seed{seed}" / "mtb_ml_models"
        binary = source_directory / f"{prefix}.bin"
        source_c = source_directory / f"{prefix}.c"
        source_h = source_directory / f"{prefix}.h"
        expected_hash = str(member["compiled_model"]["sha256"])
        _require(
            binary.is_file()
            and source_c.is_file()
            and source_h.is_file()
            and sha256_file(binary) == expected_hash,
            f"generated firmware model identity changed: {seed}",
        )
        destination_c = model_output / source_c.name
        destination_h = model_output / source_h.name
        shutil.copyfile(source_c, destination_c)
        shutil.copyfile(source_h, destination_h)
        staged.append(
            {
                "seed": seed,
                "compiled_tflite_sha256": expected_hash,
                "compiled_tflite_bytes": binary.stat().st_size,
                "c_sha256": sha256_file(destination_c),
                "h_sha256": sha256_file(destination_h),
            }
        )

    mean, std = load_normalizer(bundle)
    normalizer_h = model_output / "fastreflex_normalizer.h"
    normalizer_c = model_output / "fastreflex_normalizer.c"
    normalizer_h.write_text(
        "#ifndef FASTREFLEX_NORMALIZER_H\n"
        "#define FASTREFLEX_NORMALIZER_H\n\n"
        "extern const float fastreflex_normalizer_mean[80];\n"
        "extern const float fastreflex_normalizer_std[80];\n\n"
        "#endif\n",
        encoding="utf-8",
    )
    normalizer_c.write_text(
        '#include "fastreflex_normalizer.h"\n\n'
        + _c_float_array("fastreflex_normalizer_mean", mean)
        + "\n"
        + _c_float_array("fastreflex_normalizer_std", std),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "role": PROTOTYPE_ROLE,
        "formal_status": M31_VERDICT,
        "formal_m3_pass": False,
        "numerical_contract_pass": False,
        "source_vela_report": str(report_path.relative_to(root)),
        "source_vela_report_sha256": sha256_file(report_path),
        "normalizer_source_sha256": config["accepted_identity"]["normalizer_sha256"],
        "normalizer_c_sha256": sha256_file(normalizer_c),
        "members": staged,
    }
    manifest_path = output / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": "E84_FIRMWARE_ASSETS_STAGED",
        "directory": str(output),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "members": staged,
    }
