from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from fastreflex_e84.handoff import (
    sha256_file,
    validate_reference_bundle,
    verify_reference_handoff,
)
from fastreflex_e84.reference_runtime import (
    FEATURE_SCHEMA_SHA256,
    apply_decision,
    causal_delta,
    causal_rolling,
    extract_features,
    feature_schema,
    feature_schema_hash,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/deployment/reference_model.yaml"
BUNDLE = ROOT / "model/source/model_v2_anchor_refined_gru20_20260902"


def test_handoff_and_all_host_float_layers_pass() -> None:
    result = verify_reference_handoff(ROOT, CONFIG)
    assert (
        result["status"]
        == "REFERENCE_MODEL_HANDOFF_AND_BATCH_ONE_HOST_FLOAT_PARITY_PASS"
    )
    assert result["contract"]["files_verified"] == 18
    assert result["parity"]["status"] == "PASS"
    assert result["parity"]["canonical_execution_shape"] == [1, 20, 80]
    assert result["parity"]["continuous_parity"]["member_logits"]["absolute"] == 4e-6
    assert result["parity"]["continuous_parity"]["member_logits"]["relative"] == 0.0
    assert result["parity"]["reflex_onset_endpoints"] == [65, 90, 107]
    assert set(result["parity"]["layers"]) == {
        "raw_pelvis_imu6",
        "base_features",
        "causal_features",
        "normalized_features",
        "model_windows",
        "member_logits",
        "member_hazard_probability",
        "ensemble_hazard_probability",
        "window_endpoints",
        "threshold_crossing",
        "consecutive_threshold_count",
        "reflex_required",
        "reflex_onset",
    }
    numeric = (
        "base_features",
        "causal_features",
        "normalized_features",
        "model_windows",
        "member_logits",
        "member_hazard_probability",
        "ensemble_hazard_probability",
    )
    assert all(
        result["parity"]["layers"][name]["max_absolute_error"] == 0.0
        for name in numeric
    )
    assert result["scientific_status"] == {
        "verdict": "MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED",
        "simulation_status": "SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED",
        "release_model": False,
        "real_robot_supported": False,
    }


def test_feature_order_prefix_and_population_rolling_semantics() -> None:
    assert len(feature_schema()) == 80
    assert feature_schema_hash() == FEATURE_SCHEMA_SHA256
    assert feature_schema()[0] == "pelvis_base_accel_x"
    assert feature_schema()[9] == "pelvis_base_horizontal_gyro_norm"
    assert feature_schema()[10] == "pelvis_delta_1ms_accel_x"
    assert feature_schema()[-1] == "pelvis_causal_variance_10ms_horizontal_gyro_norm"

    scalar = np.asarray([[1.0], [3.0], [7.0]], dtype=np.float32)
    np.testing.assert_array_equal(causal_delta(scalar, 5), np.zeros_like(scalar))
    np.testing.assert_array_equal(
        causal_delta(scalar, 1),
        np.asarray([[0.0], [2.0], [4.0]], dtype=np.float32),
    )
    mean, variance = causal_rolling(scalar, 5)
    np.testing.assert_allclose(mean[:, 0], [1.0, 2.0, 11.0 / 3.0])
    np.testing.assert_allclose(
        variance[:, 0],
        [0.0, 1.0, np.var(np.asarray([1.0, 3.0, 7.0]), ddof=0)],
    )

    imu = np.zeros((12, 6), dtype=np.float32)
    imu[:, 0] = np.arange(12, dtype=np.float32)
    features = extract_features(imu)
    assert features.dtype == np.float32
    assert features.shape == (12, 80)
    np.testing.assert_array_equal(
        features[:5, 20:30], np.zeros((5, 10), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        features[:10, 30:40], np.zeros((10, 10), dtype=np.float32)
    )


def test_inclusive_threshold_and_consecutive_reset() -> None:
    probabilities = np.asarray(
        [0.99, 0.99, 0.99, 0.99, np.nextafter(0.99, 0.0)] + [0.99] * 6 + [0.0],
        dtype=np.float64,
    )
    crossing, counts, reflex, onset = apply_decision(probabilities)
    assert crossing[0]
    assert not crossing[4]
    assert counts.tolist() == [1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 0]
    assert np.flatnonzero(onset).tolist() == [9]
    assert np.flatnonzero(reflex).tolist() == [9, 10]


def _temporary_acceptance_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    bundle = tmp_path / "model/source/model_v2_anchor_refined_gru20_20260902"
    shutil.copytree(BUNDLE, bundle)
    config_path = tmp_path / "configs/deployment/reference_model.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path, config_path, bundle


def test_checksum_drift_fails_before_artifact_loading(tmp_path: Path) -> None:
    root, config, bundle = _temporary_acceptance_root(tmp_path)
    normalizer = bundle / "normalizer.json"
    normalizer.write_bytes(normalizer.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_reference_bundle(root, config)


def test_resigned_feature_order_drift_still_fails(tmp_path: Path) -> None:
    root, config_path, bundle = _temporary_acceptance_root(tmp_path)
    preprocessing_path = bundle / "preprocessing.json"
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    preprocessing["feature_order"][0], preprocessing["feature_order"][1] = (
        preprocessing["feature_order"][1],
        preprocessing["feature_order"][0],
    )
    preprocessing_path.write_text(
        json.dumps(preprocessing, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    release_manifest_path = bundle / "release_manifest.json"
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    release_manifest["files"]["preprocessing.json"] = sha256_file(preprocessing_path)
    release_manifest_path.write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["accepted_identity"]["release_manifest_sha256"] = sha256_file(
        release_manifest_path
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="feature order"):
        validate_reference_bundle(root, config_path)


def test_resigned_local_tolerance_widening_still_fails(tmp_path: Path) -> None:
    root, config_path, bundle = _temporary_acceptance_root(tmp_path)
    contract_path = bundle / "float_numerical_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["continuous_parity"]["member_logits"]["absolute"] = 5e-6
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    release_manifest_path = bundle / "release_manifest.json"
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    release_manifest["files"]["float_numerical_contract.json"] = sha256_file(
        contract_path
    )
    release_manifest_path.write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["accepted_identity"]["release_manifest_sha256"] = sha256_file(
        release_manifest_path
    )
    config["accepted_identity"]["float_numerical_contract_sha256"] = sha256_file(
        contract_path
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Research Float parity tolerances"):
        validate_reference_bundle(root, config_path)
