"""Fail-closed handoff acceptance and layered host-Float parity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from .reference_runtime import (
    BASE_FEATURES,
    CLASS_NAMES,
    FEATURE_SCHEMA_SHA256,
    MEMBER_SEEDS,
    TEMPORAL_TRANSFORMS,
    apply_decision,
    feature_schema,
    feature_schema_hash,
    load_normalizer,
    run_runtime_chain,
)


DEFAULT_CONFIG = Path("configs/deployment/reference_model.yaml")
REQUIRED_RELEASE_FILES = frozenset(
    {
        "golden_inputs/decision_probe.npz",
        "golden_inputs/runtime_chain.npz",
        "golden_manifest.json",
        "golden_outputs/decision_probe.npz",
        "golden_outputs/runtime_chain.npz",
        "label_map.json",
        "metrics.json",
        "model_manifest.json",
        "models/member_seed20260828.pt",
        "models/member_seed20260829.pt",
        "models/member_seed20260830.pt",
        "normalizer.json",
        "preprocessing.json",
        "sensor_schema.json",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolve_bundle(repository_root: Path, relative: str) -> Path:
    root = repository_root.resolve()
    bundle = (root / relative).resolve()
    if root not in bundle.parents:
        raise ValueError("reference bundle path escapes the repository")
    if not bundle.is_dir():
        raise ValueError(f"reference bundle is missing: {bundle}")
    return bundle


def validate_reference_bundle(
    repository_root: Path, config_path: Path = DEFAULT_CONFIG
) -> tuple[Path, dict[str, object], dict[str, object]]:
    """Validate identity, every file hash, and all runtime contract invariants."""
    root = repository_root.resolve()
    selected_config = (
        config_path if config_path.is_absolute() else root / config_path
    )
    config = yaml.safe_load(selected_config.read_text(encoding="utf-8"))
    _require(config.get("schema_version") == 1, "unsupported acceptance config")
    accepted = config["accepted_identity"]
    bundle = _resolve_bundle(root, str(config["bundle_path"]))
    manifest_path = bundle / "release_manifest.json"
    _require(manifest_path.is_file(), "release manifest is missing")
    _require(
        sha256_file(manifest_path) == accepted["release_manifest_sha256"],
        "release manifest does not match the deployment acceptance pin",
    )
    release_manifest = _load_json(manifest_path)
    _require(release_manifest.get("schema_version") == 1, "release schema changed")
    _require(
        release_manifest.get("release_id") == accepted["release_id"],
        "release ID changed",
    )
    files = release_manifest.get("files")
    _require(
        isinstance(files, dict) and set(files) == REQUIRED_RELEASE_FILES,
        "release manifest file set changed",
    )
    actual_files = {
        str(path.relative_to(bundle))
        for path in bundle.rglob("*")
        if path.is_file()
    }
    _require(
        actual_files == REQUIRED_RELEASE_FILES | {"release_manifest.json"},
        "release bundle contains missing or extra files",
    )
    for relative, expected in files.items():
        path = (bundle / relative).resolve()
        _require(
            bundle in path.parents and path.is_file(),
            f"invalid release file path: {relative}",
        )
        _require(
            sha256_file(path) == expected,
            f"release file checksum mismatch: {relative}",
        )

    model = _load_json(bundle / "model_manifest.json")
    sensor = _load_json(bundle / "sensor_schema.json")
    preprocessing = _load_json(bundle / "preprocessing.json")
    labels = _load_json(bundle / "label_map.json")
    metrics = _load_json(bundle / "metrics.json")
    golden = _load_json(bundle / "golden_manifest.json")
    provenance = model["provenance"]
    scientific = model["scientific_status"]
    _require(model.get("candidate_id") == accepted["release_id"], "candidate changed")
    _require(
        model.get("engineering_role") == accepted["engineering_role"],
        "engineering role changed",
    )
    _require(model.get("status") == accepted["status"], "handoff status changed")
    _require(model.get("release_model") is False, "reference became a release model")
    _require(
        provenance.get("source_repository") == accepted["research_repository"],
        "source repository changed",
    )
    for field in (
        "packaging_source_commit",
        "candidate_source_commit",
        "candidate_record_commit",
        "scientific_verdict_commit",
    ):
        _require(provenance.get(field) == accepted[field], f"{field} changed")
    _require(
        scientific.get("verdict") == accepted["scientific_verdict"],
        "scientific verdict changed",
    )
    _require(
        scientific.get("simulation_status") == accepted["simulation_status"],
        "simulation status changed",
    )
    _require(
        scientific.get("engineering_only") is True
        and scientific.get("real_robot_supported") is False
        and scientific.get("safety_certified") is False,
        "non-release safety status changed",
    )

    architecture = model["architecture"]
    _require(
        architecture.get("model_family") == "gru"
        and architecture.get("input_size") == 80
        and architecture.get("hidden_size") == 32
        and architecture.get("layers") == 1
        and architecture.get("bidirectional") is False
        and architecture.get("dropout") == 0.0
        and architecture.get("parameters") == 11_010
        and architecture.get("input_shape") == [20, 80]
        and architecture.get("input_dtype") == "float32"
        and architecture.get("member_output_shape") == [2]
        and architecture.get("member_output_dtype") == "float32",
        "model architecture contract changed",
    )
    _require(
        architecture.get("architecture_sha256")
        == accepted["architecture_sha256"],
        "architecture identity changed",
    )
    runtime = model["runtime"]
    _require(
        runtime.get("sample_rate_hz") == 1000
        and runtime.get("history_samples") == 20
        and runtime.get("replay_stride_samples") == 1
        and runtime.get("ensemble_membership") == list(MEMBER_SEEDS)
        and runtime.get("gru_hidden_state")
        == "zero_initialized_for_each_window_no_cross_window_carry"
        and runtime.get("member_softmax_compute_dtype") == "float32"
        and runtime.get("member_probability_storage_dtype") == "float64"
        and runtime.get("ensemble_mean_dtype") == "float64"
        and runtime.get("hazard_class_index") == 1
        and runtime.get("threshold") == 0.99
        and runtime.get("threshold_comparison") == "greater_than_or_equal"
        and runtime.get("persistence_samples") == 5
        and runtime.get("persistence")
        == "consecutive_samples_reset_to_zero_on_failure"
        and runtime.get("output_hold")
        == "asserted_while_current_streak_is_at_least_five_and_deasserted_immediately_on_failure"
        and runtime.get("feature_schema_sha256") == FEATURE_SCHEMA_SHA256,
        "runtime decision contract changed",
    )
    _require(
        runtime.get("feature_schema_sha256")
        == accepted["feature_schema_sha256"],
        "accepted feature schema identity changed",
    )
    members = model["ensemble_members"]
    _require(
        [member.get("seed") for member in members] == list(MEMBER_SEEDS),
        "checkpoint membership changed",
    )
    _require(
        [member.get("sha256") for member in members]
        == accepted["checkpoint_sha256"],
        "accepted checkpoint identity changed",
    )
    for member in members:
        _require(
            files.get(str(member["path"])) == member["sha256"],
            "checkpoint manifest linkage changed",
        )

    expected_channels = [
        (0, "accel_x", "m/s^2"),
        (1, "accel_y", "m/s^2"),
        (2, "accel_z", "m/s^2"),
        (3, "gyro_x", "rad/s"),
        (4, "gyro_y", "rad/s"),
        (5, "gyro_z", "rad/s"),
    ]
    _require(
        sensor.get("sensor") == "PELVIS_IMU6"
        and sensor.get("sample_rate_hz") == 1000
        and sensor.get("dtype") == "float32"
        and sensor.get("frame") == "pelvis_local"
        and sensor.get("hardware_mapping_status") == "NOT_VALIDATED"
        and [
            (item["index"], item["name"], item["unit"])
            for item in sensor["channels"]
        ]
        == expected_channels,
        "sensor schema changed",
    )
    _require(
        preprocessing.get("base_feature_order") == list(BASE_FEATURES)
        and preprocessing.get("temporal_transform_order")
        == list(TEMPORAL_TRANSFORMS)
        and preprocessing.get("feature_order") == list(feature_schema())
        and preprocessing.get("feature_schema_sha256") == FEATURE_SCHEMA_SHA256
        and feature_schema_hash() == FEATURE_SCHEMA_SHA256,
        "feature order or schema hash changed",
    )
    _require(
        preprocessing["delta"].get("unavailable_prefix") == "exact_zero"
        and preprocessing["rolling"].get("startup")
        == "use all available samples from index zero"
        and preprocessing["rolling"].get("accumulator_dtype") == "float64"
        and preprocessing["rolling"].get("variance")
        == "population variance (ddof=0), clamped to at least zero"
        and preprocessing["window"].get("shape") == [20, 80]
        and preprocessing["window"].get("endpoint") == "inclusive",
        "causal preprocessing semantics changed",
    )
    _require(
        labels.get("logit_and_softmax_class_index")
        == {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]}
        and labels.get("terrain_used_as_hazard_gate") is False,
        "class mapping or Hazard gating changed",
    )

    normalizer = _load_json(bundle / "normalizer.json")
    load_normalizer(bundle)
    _require(
        model["normalizer"].get("path") == "normalizer.json"
        and model["normalizer"].get("sha256")
        == accepted["normalizer_sha256"]
        and files.get("normalizer.json") == accepted["normalizer_sha256"],
        "accepted normalizer identity changed",
    )
    _require(
        normalizer.get("method") == "per_channel_zscore"
        and normalizer.get("feature_schema") == list(feature_schema())
        and normalizer.get("feature_schema_sha256") == FEATURE_SCHEMA_SHA256
        and normalizer.get("train_only") is True,
        "normalizer provenance or schema changed",
    )
    _require(
        metrics.get("engineering_role") == accepted["engineering_role"]
        and metrics.get("generalization_holdout", {}).get("verdict")
        == accepted["scientific_verdict"]
        and metrics.get("generalization_holdout", {}).get("simulation_status")
        == accepted["simulation_status"]
        and metrics.get("release_model") is False
        and metrics.get("real_robot_supported") is False
        and metrics.get("safety_certified") is False
        and metrics.get("quantization_authorized_by_this_handoff") is False,
        "scientific metrics status changed",
    )
    _require(
        golden.get("scientific_evidence") is False
        and golden.get("protected_holdout_access") is False
        and golden.get("source", {}).get("split") == "V2_VALIDATION"
        and golden.get("discrete_parity") == "exact",
        "golden evidence boundary changed",
    )
    return bundle, config, model


def _numeric_parity(
    actual: np.ndarray, expected: np.ndarray, absolute: float, relative: float
) -> dict[str, object]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(
            f"numeric shape/dtype mismatch: {actual.shape}/{actual.dtype} "
            f"!= {expected.shape}/{expected.dtype}"
        )
    if not np.allclose(actual, expected, atol=absolute, rtol=relative):
        maximum = float(np.max(np.abs(actual.astype(float) - expected.astype(float))))
        raise ValueError(f"numeric parity failed; maximum absolute error {maximum}")
    maximum = (
        0.0
        if not actual.size
        else float(np.max(np.abs(actual.astype(float) - expected.astype(float))))
    )
    return {
        "status": "PASS",
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_absolute_error": maximum,
    }


def _exact_parity(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or not np.array_equal(actual, expected)
    ):
        raise ValueError("exact discrete parity failed")
    return {
        "status": "PASS",
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "exact": True,
    }


def verify_reference_handoff(
    repository_root: Path, config_path: Path = DEFAULT_CONFIG
) -> dict[str, object]:
    """Prove bundle acceptance and independent host-Float runtime parity."""
    bundle, config, model = validate_reference_bundle(repository_root, config_path)
    tolerance = config["parity"]
    absolute = float(tolerance["absolute_tolerance"])
    relative = float(tolerance["relative_tolerance"])
    with np.load(
        bundle / "golden_inputs/runtime_chain.npz", allow_pickle=False
    ) as inputs:
        raw = inputs["raw_pelvis_imu6"].copy()
        source_indices = inputs["source_sample_indices"].copy()
        timestamp_us = inputs["timestamp_us"].copy()
    _require(
        raw.dtype == np.float32
        and raw.ndim == 2
        and raw.shape[1] == 6
        and np.all(np.isfinite(raw)),
        "golden raw IMU schema changed",
    )
    _require(
        source_indices.dtype == np.int64
        and np.array_equal(np.diff(source_indices), np.ones(len(raw) - 1, dtype=np.int64))
        and timestamp_us.dtype == np.int64
        and np.array_equal(
            np.diff(timestamp_us), np.full(len(raw) - 1, 1000, dtype=np.int64)
        ),
        "golden 1 kHz sample timeline changed",
    )

    chain = run_runtime_chain(bundle, model, raw)
    layers: dict[str, object] = {
        "raw_pelvis_imu6": {
            "status": "PASS",
            "shape": list(raw.shape),
            "dtype": str(raw.dtype),
            "finite": True,
            "sample_period_us": 1000,
        }
    }
    with np.load(
        bundle / "golden_outputs/runtime_chain.npz", allow_pickle=False
    ) as expected:
        for name in (
            "base_features",
            "causal_features",
            "normalized_features",
            "model_windows",
            "member_logits",
            "member_hazard_probability",
            "ensemble_hazard_probability",
        ):
            layers[name] = _numeric_parity(
                getattr(chain, name), expected[name], absolute, relative
            )
        for name in (
            "window_endpoints",
            "threshold_crossing",
            "consecutive_threshold_count",
            "reflex_required",
            "reflex_onset",
        ):
            layers[name] = _exact_parity(getattr(chain, name), expected[name])

    with np.load(
        bundle / "golden_inputs/decision_probe.npz", allow_pickle=False
    ) as inputs, np.load(
        bundle / "golden_outputs/decision_probe.npz", allow_pickle=False
    ) as expected:
        decision = apply_decision(inputs["ensemble_hazard_probability"])
        decision_names = (
            "threshold_crossing",
            "consecutive_threshold_count",
            "reflex_required",
            "reflex_onset",
        )
        decision_probe = {
            name: _exact_parity(actual, expected[name])
            for name, actual in zip(decision_names, decision)
        }

    accepted = config["accepted_identity"]
    onset_endpoints = chain.window_endpoints[chain.reflex_onset].tolist()
    return {
        "status": "REFERENCE_MODEL_HANDOFF_AND_HOST_FLOAT_PARITY_PASS",
        "candidate_id": accepted["release_id"],
        "engineering_role": accepted["engineering_role"],
        "contract": {
            "status": "PASS",
            "files_verified": len(REQUIRED_RELEASE_FILES),
            "release_manifest_sha256": accepted["release_manifest_sha256"],
            "research_release_commit": accepted["research_release_commit"],
        },
        "parity": {
            "status": "PASS",
            "absolute_tolerance": absolute,
            "relative_tolerance": relative,
            "discrete_decisions": "exact",
            "layers": layers,
            "decision_probe": decision_probe,
            "reflex_onset_endpoints": onset_endpoints,
        },
        "scientific_status": {
            "verdict": accepted["scientific_verdict"],
            "simulation_status": accepted["simulation_status"],
            "release_model": False,
            "real_robot_supported": False,
        },
        "next_milestone": "EXPORT_AND_TARGET_OPERATOR_FEASIBILITY",
    }
