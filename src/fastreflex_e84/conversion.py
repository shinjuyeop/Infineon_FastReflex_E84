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
    tf = _tensorflow()
    module, concrete = _concrete_gru(state, include_softmax=True)

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
    bundle, config, model_manifest = validate_reference_bundle(root, config_path)
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
        bundle / "golden_outputs/runtime_chain.npz", allow_pickle=False
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
    tolerance = config["parity"]
    absolute = float(tolerance["absolute_tolerance"])
    relative = float(tolerance["relative_tolerance"])
    member_parity = {
        str(seed): _numeric_parity(
            actual_logits[index], expected_logits[index], absolute, relative
        )
        for index, seed in enumerate(seeds)
    }
    parity = {
        "tolerance": {"absolute": absolute, "relative": relative},
        "member_logits": _numeric_parity(
            actual_logits, expected_logits, absolute, relative
        ),
        "member_logits_by_seed": member_parity,
        "member_hazard_probability": _numeric_parity(
            actual_probabilities, expected_probabilities, absolute, relative
        ),
        "ensemble_hazard_probability": _numeric_parity(
            actual_ensemble, expected_ensemble, absolute, relative
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
    result: dict[str, object] = {
        "status": M2_VERDICT,
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
                "operator mapping is viable, but progression is blocked by the "
                "frozen Float tolerance failure; target-specific memory "
                "configuration, INT8 parity, preprocessing, invocation overhead, "
                "and board timing also remain"
            ),
        },
        "boundary": {
            "int8_parity_completed": False,
            "firmware_started": False,
            "board_state_modified": False,
            "research_semantics_modified": False,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["result_path"] = str(result_path)
    return result
