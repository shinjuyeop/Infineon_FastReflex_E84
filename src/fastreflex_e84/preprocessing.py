"""Streaming implementation of the frozen causal Pelvis IMU6 contract."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .reference_runtime import FEATURE_DIMENSION, HISTORY_SAMPLES, load_normalizer


@dataclass(frozen=True)
class PreprocessingStep:
    base: np.ndarray
    causal: np.ndarray
    normalized: np.ndarray
    window: np.ndarray | None


class StreamingPreprocessor:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (FEATURE_DIMENSION,) or self.std.shape != (
            FEATURE_DIMENSION,
        ):
            raise ValueError("normalizer must contain 80 mean/std values")
        if np.any(self.std <= 0.0):
            raise ValueError("normalizer std must be positive")
        self._base: deque[np.ndarray] = deque(maxlen=10)
        self._normalized: deque[np.ndarray] = deque(maxlen=HISTORY_SAMPLES)

    @classmethod
    def from_bundle(cls, bundle: Path) -> StreamingPreprocessor:
        return cls(*load_normalizer(bundle))

    @staticmethod
    def _base_features(imu6: np.ndarray) -> np.ndarray:
        value = np.asarray(imu6, dtype=np.float32)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("raw Pelvis IMU sample must be finite float32[6]")
        accel = value[:3]
        gyro = value[3:]
        derived = np.asarray(
            (
                np.linalg.norm(accel),
                np.linalg.norm(gyro),
                np.linalg.norm(accel[:2]),
                np.linalg.norm(gyro[:2]),
            ),
            dtype=np.float32,
        )
        return np.concatenate((value, derived)).astype(np.float32, copy=False)

    def _delta(self, current: np.ndarray, lag: int) -> np.ndarray:
        if len(self._base) < lag:
            return np.zeros(10, dtype=np.float32)
        return (current - self._base[-lag]).astype(np.float32, copy=False)

    def _rolling(
        self, current: np.ndarray, width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        history = list(self._base)[-(width - 1) :] + [current]
        values = np.asarray(history, dtype=np.float64)
        mean = np.mean(values, axis=0, dtype=np.float64)
        variance = np.mean(values * values, axis=0, dtype=np.float64) - mean * mean
        return (
            mean.astype(np.float32),
            np.maximum(variance, 0.0).astype(np.float32),
        )

    def push(self, imu6: np.ndarray) -> PreprocessingStep:
        base = self._base_features(imu6)
        mean5, variance5 = self._rolling(base, 5)
        mean10, variance10 = self._rolling(base, 10)
        causal = np.concatenate(
            (
                base,
                self._delta(base, 1),
                self._delta(base, 5),
                self._delta(base, 10),
                mean5,
                mean10,
                variance5,
                variance10,
            )
        ).astype(np.float32, copy=False)
        normalized = ((causal - self.mean) / self.std).astype(np.float32, copy=False)
        self._base.append(base)
        self._normalized.append(normalized)
        window = None
        if len(self._normalized) == HISTORY_SAMPLES:
            window = np.asarray(self._normalized, dtype=np.float32).copy()
        return PreprocessingStep(base, causal, normalized, window)
