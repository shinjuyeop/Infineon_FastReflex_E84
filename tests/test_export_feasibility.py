from __future__ import annotations

import json
from pathlib import Path

from fastreflex_e84.conversion import (
    M21_VERDICT,
    M31_VERDICT,
    M3_VERDICT,
    evaluate_export_feasibility,
    evaluate_int8_recovery,
    evaluate_int8_quantization,
)
from fastreflex_e84.handoff import sha256_file


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
        member["repeated_conversion_byte_identical"] for member in formal["members"]
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
        len(member["quantization_summary"]["learned_matrix_weight_tensors"]) == 3
        for member in formal["members"]
    )

    assert result["int8_numerical_contract"]["status"] == "FAIL"
    assert result["int8_numerical_contract"]["discrete_status"] == "PASS"
    assert (
        result["int8_numerical_contract"]["checks"][
            "member_probability_maximum_absolute_error"
        ]["status"]
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
        result["alternatives"]["unclipped_full_train_range_npu_softmax"]["parity"][
            "threshold_crossing"
        ]["mismatch_count"]
        == 22
    )

    for memory_mode in ("Shared_Sram", "Sram_Only"):
        rows = result["vela"]["members"][memory_mode]
        assert set(rows) == {"20260828", "20260829", "20260830"}
        assert all(row["cpu_operators"] == 0 for row in rows.values())
        assert all(row["npu_operators"] == 192 for row in rows.values())


def test_int8_recurrent_localization_and_ptq_recovery_remain_fail_closed(
    tmp_path: Path,
) -> None:
    result = evaluate_int8_recovery(ROOT, CONFIG, tmp_path)

    assert result["status"] == M31_VERDICT
    baseline = result["m3_baseline_reproduction"]
    assert baseline["artifact_hashes_match"] is True
    assert baseline["contract"]["continuous_status"] == "FAIL"
    assert baseline["contract"]["discrete_status"] == "PASS"
    assert result["localization"]["worst_window_indices_by_seed"] == {
        "20260828": 107,
        "20260829": 105,
        "20260830": 111,
    }
    worst_traces = result["localization"]["baseline_traces"][:3]
    assert [
        trace["trace"]["first_hidden_state_material_timestep"] for trace in worst_traces
    ] == [2, 3, 1]
    assert all(
        trace["trace"]["input_projection_is_material_before_recurrence"]
        for trace in worst_traces
    )

    selection = result["selection"]
    assert selection["name"] == "two_blocks_per_gate_16"
    assert selection["golden_used"] is False
    assert selection["byte_deterministic_all_members"] is True
    assert selection["contract"]["continuous_status"] == "FAIL"
    assert selection["contract"]["discrete_status"] == "PASS"
    assert (
        selection["contract"]["checks"]["member_probability_maximum_absolute_error"][
            "observed_maximum_across_members"
        ]
        == 0.2613510489463806
    )
    assert (
        selection["contract"]["checks"]["ensemble_probability_maximum_absolute_error"][
            "observed"
        ]
        == 0.08326796442270279
    )

    for memory_mode in ("Shared_Sram", "Sram_Only"):
        rows = result["vela"]["members"][memory_mode]
        assert all(row["cpu_operators"] == 0 for row in rows.values())
        assert all(row["npu_operators"] == 472 for row in rows.values())
    assert result["formal_m3_rerun"]["performed"] is False
    assert result["root_cause_assessment"]["research_intervention_required"]
    assert result["boundary"]["m4_authorized"] is False
    assert result["boundary"]["board_state_modified"] is False


def test_frozen_hil_prototype_identity_and_formal_separation() -> None:
    prototype = (
        ROOT
        / "model/prototype"
        / "model_v2_anchor_refined_gru20_int8_16ch_hil_prototype"
    )
    manifest = json.loads(
        (prototype / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["role"] == "NON_RELEASE_HIL_PATH_PROTOTYPE"
    assert manifest["formal_status"] == M31_VERDICT
    assert manifest["formal_m3_pass"] is False
    assert manifest["numerical_contract_pass"] is False
    assert manifest["scientific_release"] is False
    assert manifest["real_robot_supported"] is False
    assert manifest["production_ready"] is False
    assert manifest["safety_certified"] is False
    assert manifest["hil_path_only"] is True
    assert manifest["m4_authorized"] is False
    assert manifest["runtime_contract"]["ensemble_order"] == [
        20260828,
        20260829,
        20260830,
    ]
    assert manifest["runtime_contract"]["input_shape"] == [1, 20, 80]
    assert manifest["canonical_golden_parity"]["numerical_contract"] == "FAIL"
    assert all(manifest["canonical_golden_parity"]["exact"].values())
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert path.parent == prototype / "models"
        assert sha256_file(path) == artifact["sha256"]

    formal_hashes = {
        20260828: "d6726d65a6bb63ca564b57b528112ab6e6db5a0e1b77b87cdae08c4ef363aec1",
        20260829: "c793df173ecf73c80f05bae6dbb2061a552f2e4448fff02025892c547a19d3c3",
        20260830: "f4538854a092d4d12a8909f06d2a01156c8097f54c4518b31d9df8a82ad1a854",
    }
    for seed, expected in formal_hashes.items():
        path = (
            ROOT
            / "model/quantized/formal"
            / f"member_seed{seed}_int8_probability.tflite"
        )
        assert sha256_file(path) == expected


def test_target_vela_evidence_is_target_specific_and_non_release() -> None:
    report = json.loads(
        (ROOT / "reports/e84_target_vela_compilation.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "E84_TARGET_VELA_COMPILATION_PASS"
    assert report["role"] == "NON_RELEASE_HIL_PATH_PROTOTYPE"
    assert report["formal_status"] == M31_VERDICT
    assert report["formal_m3_pass"] is False
    assert report["numerical_contract_pass"] is False
    assert report["target"]["accelerator_config"] == "ethos-u55-128"
    assert report["target"]["system_config"] == (
        "PSE8x_U55_400MHz_SOCMEM_200MHz_QUAD_XIP"
    )
    assert report["target"]["memory_mode"] == "Sram_Only"
    assert report["toolchain"]["vela_version"] == "4.3.0"

    members = report["members"]
    assert [member["seed"] for member in members] == [
        20260828,
        20260829,
        20260830,
    ]
    assert all(member["cpu_operators"] == 0 for member in members)
    assert all(member["npu_operators"] == 472 for member in members)
    assert all(member["macs_per_inference"] == 212676 for member in members)
    assert [member["command_stream_bytes"] for member in members] == [
        33776,
        33804,
        33820,
    ]
    total = report["three_member_sequential_summary"]
    assert total["compiled_model_bytes"] == 352592
    assert total["total_cycles"] == 114231
    assert total["tensor_arena_bytes_if_reused"] == 16384
    assert total["cpu_operators"] == 0
    assert report["boundary"]["board_execution_measured"] is False
    assert report["boundary"]["m4_authorized"] is False

    vela_config = (ROOT / "configs/vela/pse84_u55.ini").read_text(
        encoding="utf-8"
    )
    assert "[System_Config.PSE8x_U55_400MHz_SOCMEM_200MHz_QUAD_XIP]" in vela_config
    assert "Sram_clock_scale=0.47" in vela_config
    assert "[Memory_Mode.Sram_Only]" in vela_config
