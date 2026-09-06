from pathlib import Path

from scripts.generate_v1_result_assets import generate


def test_locked_v1_assets_regenerate(tmp_path: Path):
    generate(None, tmp_path)
    expected = {
        "README.md",
        "tables/clean_metrics.csv",
        "tables/q1_contrasts.csv",
        "tables/q2_pareto.csv",
        "tables/q3_condition_rankings.csv",
        "tables/cost.csv",
        "tables/calibration_transfer.csv",
        "tables/table_clean_metrics.tex",
        "tables/table_q2_pareto.tex",
        "figures/fig_clean_rocch_eer.tex",
        "figures/fig_q2_eer_latency_pareto.tex",
        "figures/fig_q3_conditionwise_eer.tex",
        "figures/fig_calibration_transfer.tex",
    }
    actual = {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()}
    assert actual == expected
    assert "C4" in (tmp_path / "README.md").read_text(encoding="utf-8")
    committed = Path(__file__).resolve().parents[1] / "results" / "v1"
    for relative in expected:
        assert (tmp_path / relative).read_bytes() == (committed / relative).read_bytes()
