from __future__ import annotations

from pathlib import Path

from fastreflex_e84.conversion import (
    M21_VERDICT,
    M3_VERDICT,
    evaluate_export_feasibility,
    evaluate_int8_quantization,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/deployment/reference_model.yaml"


def test_float_export_parity_and_u55_operator_mapping(tmp_path: Path) -> None:
    result = evaluate_export_feasibility(ROOT, CONFIG, tmp_path)

    assert result["status"] == M21_VERDICT
    exported = result["float_export"]
    assert len(exported["members"]) == 3
    assert exported["parity"]["status"] == "PASS"
    assert all(
        member["tolerance_violation_count"] == 0
        for member in exported["parity"]["member_logits_by_seed"].values()
    )
    assert exported["parity"]["threshold_crossing"]["exact"]
    assert exported["parity"]["reflex_required"]["exact"]

    float_mapping = result["vela"]["float_shared_sram"]
    assert float_mapping["cpu_operators"] == 301
    assert float_mapping["npu_operators"] == 0

    for name in ("int8_shared_sram", "int8_sram_only"):
        mapping = result["vela"][name]
        assert mapping["cpu_operators"] == 0
        assert mapping["npu_operators"] > 0
    assert result["int8_operator_probe"]["graph"]["operators"]["SOFTMAX"] == 1

    assert result["boundary"] == {
        "int8_parity_completed": False,
        "m3_authorized": True,
        "firmware_started": False,
        "board_state_modified": False,
        "research_semantics_modified": False,
    }


def test_formal_int8_characterization_fails_closed_on_probability_instability(
    tmp_path: Path,
) -> None:
    result = evaluate_int8_quantization(ROOT, CONFIG, tmp_path)

    assert result["status"] == M3_VERDICT
    assert result["calibration"]["run_count"] == 442
    assert result["calibration"]["window_count"] == 2597
    assert result["calibration"]["protected_holdout_access"] is False
    assert result["calibration"]["deployment_range_policy"]["bound"] == (
        4.132843623161313
    )

    formal = result["selected_representation"]["formal"]
    assert [member["seed"] for member in formal["members"]] == [
        20260828,
        20260829,
        20260830,
    ]
    assert all(
        member["repeated_conversion_byte_identical"]
        for member in formal["members"]
    )
    assert all(
        member["io"]["input"]["quantization"]
        == {"scale": 0.03241445869207382, "zero_point": 0}
        for member in formal["members"]
    )
    assert all(
        member["io"]["output"]["quantization"]
        == {"scale": 0.00390625, "zero_point": -128}
        for member in formal["members"]
    )
    assert all(
        len(member["quantization_summary"]["learned_matrix_weight_tensors"])
        == 3
        for member in formal["members"]
    )

    assert result["int8_numerical_contract"]["status"] == "FAIL"
    assert result["int8_numerical_contract"]["discrete_status"] == "PASS"
    assert (
        result["int8_numerical_contract"]["checks"]
        ["member_probability_maximum_absolute_error"]["status"]
        == "FAIL"
    )
    for name in (
        "threshold_crossing",
        "consecutive_threshold_count",
        "reflex_required",
        "reflex_onset",
    ):
        assert formal["parity"][name]["exact"]

    assert result["threshold_sensitivity"]["int8_onset_endpoints"] == [65, 90, 107]
    assert result["boundary"]["m4_authorized"] is False
    assert result["boundary"]["board_state_modified"] is False
    assert (
        result["alternatives"]["unclipped_full_train_range_npu_softmax"]
        ["parity"]["threshold_crossing"]["mismatch_count"]
        == 22
    )

    for memory_mode in ("Shared_Sram", "Sram_Only"):
        rows = result["vela"]["members"][memory_mode]
        assert set(rows) == {"20260828", "20260829", "20260830"}
        assert all(row["cpu_operators"] == 0 for row in rows.values())
        assert all(row["npu_operators"] == 192 for row in rows.values())
