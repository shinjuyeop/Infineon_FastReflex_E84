from __future__ import annotations

from pathlib import Path

from fastreflex_e84.conversion import (
    M2_VERDICT,
    evaluate_export_feasibility,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/deployment/reference_model.yaml"


def test_float_export_parity_and_u55_operator_mapping(tmp_path: Path) -> None:
    result = evaluate_export_feasibility(ROOT, CONFIG, tmp_path)

    assert result["status"] == M2_VERDICT
    exported = result["float_export"]
    assert len(exported["members"]) == 3
    assert exported["parity"]["status"] == "FAIL"
    assert (
        exported["parity"]["member_logits_by_seed"]["20260829"][
            "tolerance_violation_count"
        ]
        > 0
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
        "firmware_started": False,
        "board_state_modified": False,
        "research_semantics_modified": False,
    }
