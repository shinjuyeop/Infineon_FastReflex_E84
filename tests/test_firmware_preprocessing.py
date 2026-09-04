from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fastreflex_e84.reference_runtime import load_normalizer


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "model/source/model_v2_anchor_refined_gru20_20260902"
FIRMWARE = ROOT / "firmware/fastreflex_e84/proj_cm55"


class FirmwarePreprocessor(ctypes.Structure):
    _fields_ = [
        ("base_history", (ctypes.c_float * 10) * 10),
        ("base_count", ctypes.c_uint32),
        ("base_next", ctypes.c_uint32),
        ("normalized_ring", (ctypes.c_float * 80) * 20),
        ("quantized_ring", (ctypes.c_int8 * 80) * 20),
        ("normalized_count", ctypes.c_uint32),
        ("normalized_next", ctypes.c_uint32),
    ]


def _c_array(name: str, values: np.ndarray) -> str:
    encoded = ",".join(float(np.float32(value)).hex() + "f" for value in values)
    return f"const float {name}[80]={{ {encoded} }};\n"


def test_firmware_c_preprocessing_matches_frozen_golden(tmp_path: Path) -> None:
    compiler = shutil.which("gcc")
    if compiler is None:
        pytest.skip("host gcc is unavailable")
    mean, std = load_normalizer(BUNDLE)
    (tmp_path / "fastreflex_normalizer.h").write_text(
        "extern const float fastreflex_normalizer_mean[80];\n"
        "extern const float fastreflex_normalizer_std[80];\n",
        encoding="utf-8",
    )
    normalizer_source = tmp_path / "normalizer.c"
    normalizer_source.write_text(
        '#include "fastreflex_normalizer.h"\n'
        + _c_array("fastreflex_normalizer_mean", mean)
        + _c_array("fastreflex_normalizer_std", std),
        encoding="utf-8",
    )
    library_path = tmp_path / "libfastreflex_preprocessing.so"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-shared",
            "-fPIC",
            "-O2",
            "-I",
            str(tmp_path),
            "-I",
            str(FIRMWARE),
            str(FIRMWARE / "fastreflex_preprocessing.c"),
            str(normalizer_source),
            "-lm",
            "-o",
            str(library_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(library_path))
    float_pointer = ctypes.POINTER(ctypes.c_float)
    library.fastreflex_preprocessor_reset.argtypes = [
        ctypes.POINTER(FirmwarePreprocessor)
    ]
    library.fastreflex_extract_causal.argtypes = [
        ctypes.POINTER(FirmwarePreprocessor),
        float_pointer,
        float_pointer,
    ]
    library.fastreflex_normalize_and_push.argtypes = [
        ctypes.POINTER(FirmwarePreprocessor),
        float_pointer,
        float_pointer,
    ]
    library.fastreflex_normalize_and_push.restype = ctypes.c_bool
    library.fastreflex_copy_quantized_window.argtypes = [
        ctypes.POINTER(FirmwarePreprocessor),
        ctypes.POINTER(ctypes.c_int8),
    ]

    with np.load(
        BUNDLE / "golden_inputs/runtime_chain.npz", allow_pickle=False
    ) as data:
        raw = data["raw_pelvis_imu6"].astype(np.float32)
    with np.load(
        BUNDLE / "golden_outputs/deployment_runtime_chain.npz", allow_pickle=False
    ) as data:
        golden = {name: data[name].copy() for name in data.files}

    state = FirmwarePreprocessor()
    library.fastreflex_preprocessor_reset(ctypes.byref(state))
    causal_rows: list[np.ndarray] = []
    normalized_rows: list[np.ndarray] = []
    windows: list[np.ndarray] = []
    quantized_windows: list[np.ndarray] = []
    for sample in raw:
        causal = np.empty(80, dtype=np.float32)
        window = np.empty((20, 80), dtype=np.float32)
        library.fastreflex_extract_causal(
            ctypes.byref(state),
            sample.ctypes.data_as(float_pointer),
            causal.ctypes.data_as(float_pointer),
        )
        ready = library.fastreflex_normalize_and_push(
            ctypes.byref(state),
            causal.ctypes.data_as(float_pointer),
            window.ctypes.data_as(float_pointer),
        )
        row_index = (state.normalized_next + 19) % 20
        normalized = np.ctypeslib.as_array(state.normalized_ring[row_index]).copy()
        causal_rows.append(causal.copy())
        normalized_rows.append(normalized)
        if ready:
            windows.append(window.copy())
            quantized_window = np.empty((20, 80), dtype=np.int8)
            library.fastreflex_copy_quantized_window(
                ctypes.byref(state),
                quantized_window.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            )
            quantized_windows.append(quantized_window.copy())

    causal_actual = np.stack(causal_rows)
    normalized_actual = np.stack(normalized_rows)
    window_actual = np.stack(windows)
    quantized_window_actual = np.stack(quantized_windows)
    quantized_window_expected = np.clip(
        np.rint(golden["model_windows"] / np.float32(0.03241445869207382)),
        -128,
        127,
    ).astype(np.int8)
    assert np.max(np.abs(causal_actual[:, :10] - golden["base_features"])) <= 1.0e-6
    assert np.max(np.abs(causal_actual - golden["causal_features"])) <= 1.0e-6
    assert np.max(np.abs(normalized_actual - golden["normalized_features"])) <= 1.5e-6
    assert np.max(np.abs(window_actual - golden["model_windows"])) <= 1.5e-6
    np.testing.assert_array_equal(quantized_window_actual, quantized_window_expected)
