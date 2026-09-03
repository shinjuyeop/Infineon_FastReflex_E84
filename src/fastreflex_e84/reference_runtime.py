"""Independent host-Float implementation of the frozen reference runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


BASE_FEATURES = (
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_norm",
    "gyro_norm",
    "horizontal_accel_norm",
    "horizontal_gyro_norm",
)
TEMPORAL_TRANSFORMS = (
    "base",
    "delta_1ms",
    "delta_5ms",
    "delta_10ms",
    "causal_mean_5ms",
    "causal_mean_10ms",
    "causal_variance_5ms",
    "causal_variance_10ms",
)
FEATURE_DIMENSION = 80
HISTORY_SAMPLES = 20
THRESHOLD = 0.99
PERSISTENCE_SAMPLES = 5
FEATURE_SCHEMA_SHA256 = (
    "fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb"
)
CLASS_NAMES = ("NORMAL", "HAZARD_REFLEX_REQUIRED")
MEMBER_SEEDS = (20260828, 20260829, 20260830)


@dataclass(frozen=True)
class RuntimeChain:
    base_features: np.ndarray
    causal_features: np.ndarray
    normalized_features: np.ndarray
    window_endpoints: np.ndarray
    model_windows: np.ndarray
    member_logits: np.ndarray
    member_hazard_probability: np.ndarray
    ensemble_hazard_probability: np.ndarray
    threshold_crossing: np.ndarray
    consecutive_threshold_count: np.ndarray
    reflex_required: np.ndarray
    reflex_onset: np.ndarray


class ReferenceGRU(nn.Module):
    """Exact one-layer unidirectional GRU32 plus Linear(32,2)."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=FEATURE_DIMENSION,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.classifier = nn.Linear(32, len(CLASS_NAMES))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(inputs)
        return self.classifier(hidden[-1])


def feature_schema() -> tuple[str, ...]:
    return tuple(
        f"pelvis_{transform}_{name}"
        for transform in TEMPORAL_TRANSFORMS
        for name in BASE_FEATURES
    )


def feature_schema_hash(schema: tuple[str, ...] | None = None) -> str:
    selected = feature_schema() if schema is None else schema
    return hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def extract_base_features(imu6: np.ndarray) -> np.ndarray:
    values = np.asarray(imu6, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Pelvis IMU input must have shape [samples,6]")
    if not np.all(np.isfinite(values)):
        raise ValueError("Pelvis IMU input contains nonfinite values")
    accel, gyro = values[:, :3], values[:, 3:]
    derived = np.column_stack(
        (
            np.linalg.norm(accel, axis=1),
            np.linalg.norm(gyro, axis=1),
            np.linalg.norm(accel[:, :2], axis=1),
            np.linalg.norm(gyro[:, :2], axis=1),
        )
    ).astype(np.float32)
    return np.concatenate((values, derived), axis=1)


def causal_delta(values: np.ndarray, lag_samples: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or lag_samples <= 0:
        raise ValueError("causal delta expects [samples,features] and positive lag")
    result = np.zeros_like(array)
    if lag_samples < len(array):
        result[lag_samples:] = array[lag_samples:] - array[:-lag_samples]
    return result


def causal_rolling(
    values: np.ndarray, width_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or width_samples <= 0:
        raise ValueError("causal rolling expects [samples,features] and positive width")
    prefix = np.vstack((np.zeros((1, array.shape[1])), np.cumsum(array, axis=0)))
    square = np.vstack(
        (np.zeros((1, array.shape[1])), np.cumsum(array * array, axis=0))
    )
    ends = np.arange(1, len(array) + 1)
    starts = np.maximum(0, ends - int(width_samples))
    counts = (ends - starts)[:, None]
    mean = (prefix[ends] - prefix[starts]) / counts
    variance = (square[ends] - square[starts]) / counts - mean * mean
    return mean.astype(np.float32), np.maximum(variance, 0.0).astype(np.float32)


def extract_features(imu6: np.ndarray) -> np.ndarray:
    base = extract_base_features(imu6)
    mean5, variance5 = causal_rolling(base, 5)
    mean10, variance10 = causal_rolling(base, 10)
    result = np.concatenate(
        (
            base,
            causal_delta(base, 1),
            causal_delta(base, 5),
            causal_delta(base, 10),
            mean5,
            mean10,
            variance5,
            variance10,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    if result.shape != (len(base), FEATURE_DIMENSION):
        raise ValueError("Hazard features must have shape [samples,80]")
    if not np.all(np.isfinite(result)):
        raise ValueError("Hazard feature tensor is nonfinite")
    return result


def load_normalizer(bundle: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads((bundle / "normalizer.json").read_text(encoding="utf-8"))
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if (
        mean.shape != (FEATURE_DIMENSION,)
        or std.shape != (FEATURE_DIMENSION,)
        or np.any(std <= 0.0)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
    ):
        raise ValueError("normalizer tensor contract changed")
    return mean, std


def normalize_features(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIMENSION:
        raise ValueError("normalization expects [samples,80]")
    return ((values - mean) / std).astype(np.float32, copy=False)


def build_windows(normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(normalized, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIMENSION:
        raise ValueError("window construction expects [samples,80]")
    endpoints = np.arange(HISTORY_SAMPLES - 1, len(values), dtype=np.int64)
    offsets = np.arange(HISTORY_SAMPLES - 1, -1, -1, dtype=np.int64)
    windows = values[endpoints[:, None] - offsets[None, :]].astype(
        np.float32, copy=False
    )
    return endpoints, windows


def load_ensemble(
    bundle: Path, model_manifest: dict[str, object]
) -> tuple[ReferenceGRU, ...]:
    records = model_manifest["ensemble_members"]
    if [int(record["seed"]) for record in records] != list(MEMBER_SEEDS):
        raise ValueError("ensemble membership or order changed")
    models: list[ReferenceGRU] = []
    expected_state = {
        "gru.weight_ih_l0": (96, 80),
        "gru.weight_hh_l0": (96, 32),
        "gru.bias_ih_l0": (96,),
        "gru.bias_hh_l0": (96,),
        "classifier.weight": (2, 32),
        "classifier.bias": (2,),
    }
    for record in records:
        checkpoint = torch.load(
            bundle / str(record["path"]), map_location="cpu", weights_only=True
        )
        if (
            checkpoint.get("format") != "fastreflex_raw_imu_baseline"
            or checkpoint.get("family") != "gru"
            or int(checkpoint.get("window_samples", -1)) != HISTORY_SAMPLES
            or int(checkpoint.get("input_channels", -1)) != FEATURE_DIMENSION
            or checkpoint.get("class_names") != list(CLASS_NAMES)
            or int(checkpoint.get("seed", -1)) != int(record["seed"])
        ):
            raise ValueError("checkpoint metadata contract changed")
        state = checkpoint.get("state_dict")
        if not isinstance(state, dict) or set(state) != set(expected_state):
            raise ValueError("checkpoint state keys changed")
        if any(
            tuple(state[name].shape) != shape or state[name].dtype != torch.float32
            for name, shape in expected_state.items()
        ):
            raise ValueError("checkpoint state shape or dtype changed")
        model = ReferenceGRU()
        model.load_state_dict(state, strict=True)
        model.eval()
        if sum(parameter.numel() for parameter in model.parameters()) != 11_010:
            raise ValueError("reference model parameter count changed")
        models.append(model)
    return tuple(models)


def apply_decision(
    probabilities: np.ndarray,
    threshold: float = THRESHOLD,
    persistence_samples: int = PERSISTENCE_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or persistence_samples <= 0:
        raise ValueError("invalid Hazard decision inputs")
    crossing = values >= threshold
    counts = np.zeros(len(values), dtype=np.int64)
    reflex = np.zeros(len(values), dtype=bool)
    onset = np.zeros(len(values), dtype=bool)
    count = 0
    previous = False
    for index, passes in enumerate(crossing):
        count = count + 1 if bool(passes) else 0
        current = count >= persistence_samples
        counts[index] = count
        reflex[index] = current
        onset[index] = current and not previous
        previous = current
    return crossing, counts, reflex, onset


def run_runtime_chain(
    bundle: Path, model_manifest: dict[str, object], raw_imu6: np.ndarray
) -> RuntimeChain:
    base = extract_base_features(raw_imu6)
    features = extract_features(raw_imu6)
    mean, std = load_normalizer(bundle)
    normalized = normalize_features(features, mean, std)
    endpoints, windows = build_windows(normalized)
    models = load_ensemble(bundle, model_manifest)
    tensor = torch.from_numpy(windows)
    with torch.no_grad():
        # Match the handoff's explicit stateless GRU backend warm-up, then
        # derive logits and probabilities from the same stable forward pass.
        for model in models:
            model(tensor)
        member_outputs = [model(tensor) for model in models]
        logits = np.stack(
            [value.cpu().numpy() for value in member_outputs]
        ).astype(np.float32, copy=False)
        member_probability = np.stack(
            [
                torch.softmax(value, dim=1)[:, 1].cpu().numpy()
                for value in member_outputs
            ]
        ).astype(np.float64)
    ensemble = np.mean(member_probability, axis=0)
    crossing, counts, reflex, onset = apply_decision(ensemble)
    return RuntimeChain(
        base_features=base,
        causal_features=features,
        normalized_features=normalized,
        window_endpoints=endpoints,
        model_windows=windows,
        member_logits=logits,
        member_hazard_probability=member_probability,
        ensemble_hazard_probability=ensemble,
        threshold_crossing=crossing,
        consecutive_threshold_count=counts,
        reflex_required=reflex,
        reflex_onset=onset,
    )


if feature_schema_hash() != FEATURE_SCHEMA_SHA256:
    raise RuntimeError("host feature order differs from the frozen schema")
