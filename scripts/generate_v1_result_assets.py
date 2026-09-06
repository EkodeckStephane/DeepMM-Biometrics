#!/usr/bin/env python3
"""Regenerate all V1 Q1--Q3 tables and PGFPlots figures from locked evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from deepmm.evaluation.v1_final_results_lock import (
    V1_FINAL_ARTIFACT_ID,
    V1_FINAL_ARTIFACT_ZIP_SHA256,
    V1_FINAL_RESULTS_LOCK_SHA256,
    V1_FINAL_RUN_ATTEMPT,
    V1_FINAL_RUN_ID,
    load_v1_final_results_lock,
)


SYSTEMS = ("U-FP", "U-FV", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3S")
FUSION_SYSTEMS = ("C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3S")
CONDITION_LABELS = {
    "clean": "Clean",
    "fingerprint-blur-1": "FP-B1",
    "fingerprint-blur-2": "FP-B2",
    "fingerprint-blur-3": "FP-B3",
    "fingerprint-contrast-1": "FP-C1",
    "fingerprint-contrast-2": "FP-C2",
    "fingerprint-contrast-3": "FP-C3",
    "finger_vein-blur-1": "FV-B1",
    "finger_vein-blur-2": "FV-B2",
    "finger_vein-blur-3": "FV-B3",
    "finger_vein-contrast-1": "FV-C1",
    "finger_vein-contrast-2": "FV-C2",
    "finger_vein-contrast-3": "FV-C3",
    "missing-fingerprint": "No-FP",
    "missing-finger-vein": "No-FV",
}


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _latex_table(path: Path, columns: str, header: str, rows: Iterable[str], caption: str, label: str) -> None:
    body = "\n".join(rows)
    path.write_text(
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{columns}}}\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n",
        encoding="utf-8",
    )


def _clean_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    clean = result["conditions"]["clean"]["systems"]
    for system in SYSTEMS:
        metrics = clean[system]["primary_clean_calibrator"]
        rows.append({
            "system_id": system,
            "rocch_eer": metrics["rocch_eer"],
            "empirical_eer": metrics["empirical_eer"],
            "auc": metrics["auc"],
            "tar_far_0.1": metrics["tar_at_far"]["0.1"]["tar"],
            "tar_far_0.01": metrics["tar_at_far"]["0.01"]["tar"],
            "tar_far_0.001": metrics["tar_at_far"]["0.001"]["tar"],
            "cllr": metrics["cllr"],
            "cllr_cal": metrics["cllr_cal"],
            "brier": metrics["brier"],
            "nll": metrics["nll"],
            "ece": metrics["ece"],
        })
    return rows


def _write_figures(result: dict[str, Any], figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    clean = _clean_rows(result)
    clean_coords = " ".join(f"({r['system_id']},{100*r['rocch_eer']:.6f})" for r in clean)
    (figures / "fig_clean_rocch_eer.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n"
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[ybar, width=0.95\\textwidth, height=6cm, ymin=0, "
        "ylabel={ROCCH-EER (\\%)}, symbolic x coords={" + ",".join(SYSTEMS) + "}, "
        "xtick=data, x tick label style={rotate=45,anchor=east}, nodes near coords, "
        "nodes near coords style={font=\\scriptsize}, grid=major]\n"
        f"\\addplot coordinates {{{clean_coords}}};\n"
        "\\end{axis}\n\\end{tikzpicture}\n"
        "\\caption{Clean-condition discrimination on the locked V1 final trials. Lower is better.}\n"
        "\\label{fig:v1-clean-eer}\n\\end{figure*}\n",
        encoding="utf-8",
    )

    q2 = result["q2"]["rows"]
    frontier = [r for r in q2 if r["pareto_non_dominated"]]
    dominated = [r for r in q2 if not r["pareto_non_dominated"]]
    def coords(rows: list[dict[str, Any]]) -> str:
        return " ".join(
            f"({r['fusion_latency_median_ms']:.9f},{100*r['clean_rocch_eer']:.6f}) [{r['system_id']}]"
            for r in rows
        )
    (figures / "fig_q2_eer_latency_pareto.tex").write_text(
        "\\begin{figure}[t]\n\\centering\n\\begin{tikzpicture}\n"
        "\\begin{axis}[width=0.98\\columnwidth,height=6.2cm,xmode=log,"
        "xlabel={Fusion-only median latency (ms; log scale)},ylabel={Clean ROCCH-EER (\\%)},"
        "grid=major,legend pos=north east,point meta=explicit symbolic,nodes near coords,"
        "nodes near coords style={font=\\scriptsize,anchor=south}]\n"
        f"\\addplot+[only marks,mark=*,mark size=2.4pt] coordinates {{{coords(frontier)}}};\n"
        "\\addlegendentry{Point Pareto frontier}\n"
        f"\\addplot+[only marks,mark=x,mark size=2.4pt] coordinates {{{coords(dominated)}}};\n"
        "\\addlegendentry{Dominated}\n"
        "\\end{axis}\n\\end{tikzpicture}\n"
        "\\caption{Two visible Q2 dimensions. Pareto membership is computed from all four locked dimensions, including robustness loss and $C_{llr}$.}\n"
        "\\label{fig:v1-q2-pareto}\n\\end{figure}\n",
        encoding="utf-8",
    )

    ordered_conditions = list(result["conditions"])
    plots = []
    for system in FUSION_SYSTEMS:
        points = " ".join(
            f"({CONDITION_LABELS[c]},{100*result['conditions'][c]['systems'][system]['primary_clean_calibrator']['rocch_eer']:.6f})"
            for c in ordered_conditions
        )
        plots.append(f"\\addplot coordinates {{{points}}};\n\\addlegendentry{{{system}}}")
    (figures / "fig_q3_conditionwise_eer.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n\\begin{tikzpicture}\n"
        "\\begin{axis}[width=0.98\\textwidth,height=8cm,ylabel={ROCCH-EER (\\%)},"
        "symbolic x coords={" + ",".join(CONDITION_LABELS[c] for c in ordered_conditions) + "},"
        "xtick=data,x tick label style={rotate=55,anchor=east,font=\\scriptsize},"
        "legend columns=4,legend style={at={(0.5,1.02)},anchor=south,font=\\scriptsize},"
        "grid=major,mark size=1.4pt]\n"
        + "\n".join(plots)
        + "\n\\end{axis}\n\\end{tikzpicture}\n"
        "\\caption{Condition-wise fusion-system discrimination using the clean calibrator transfer policy. Missing-modality ties follow the frozen M0 fallback.}\n"
        "\\label{fig:v1-q3-conditionwise}\n\\end{figure*}\n",
        encoding="utf-8",
    )

    primary, secondary = [], []
    for condition in ordered_conditions:
        systems = result["conditions"][condition]["systems"]
        primary.append(np.mean([systems[s]["primary_clean_calibrator"]["cllr"] for s in FUSION_SYSTEMS]))
        secondary.append(np.mean([systems[s]["secondary_condition_calibrator"]["cllr"] for s in FUSION_SYSTEMS]))
    primary_coords = " ".join(f"({CONDITION_LABELS[c]},{v:.6f})" for c, v in zip(ordered_conditions, primary))
    secondary_coords = " ".join(f"({CONDITION_LABELS[c]},{v:.6f})" for c, v in zip(ordered_conditions, secondary))
    (figures / "fig_calibration_transfer.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n\\begin{tikzpicture}\n"
        "\\begin{axis}[width=0.98\\textwidth,height=7cm,ylabel={Mean $C_{llr}$},"
        "symbolic x coords={" + ",".join(CONDITION_LABELS[c] for c in ordered_conditions) + "},"
        "xtick=data,x tick label style={rotate=55,anchor=east,font=\\scriptsize},grid=major,"
        "legend style={at={(0.5,1.02)},anchor=south,legend columns=2}]\n"
        f"\\addplot coordinates {{{primary_coords}}};\n\\addlegendentry{{Clean calibrator transferred}}\n"
        f"\\addplot coordinates {{{secondary_coords}}};\n\\addlegendentry{{Matching-condition recalibration}}\n"
        "\\end{axis}\n\\end{tikzpicture}\n"
        "\\caption{Calibration robustness over the eight fusion systems. Matching-condition recalibration is diagnostic and does not replace the primary clean-calibrator transfer result.}\n"
        "\\label{fig:v1-calibration-transfer}\n\\end{figure*}\n",
        encoding="utf-8",
    )


def generate(evidence: Path | None, output: Path) -> None:
    result = load_v1_final_results_lock(evidence)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    clean_rows = _clean_rows(result)
    _write_csv(tables / "clean_metrics.csv", clean_rows, list(clean_rows[0]))
    _write_csv(tables / "q1_contrasts.csv", result["q1"]["contrasts"], ["contrast", "system_a", "system_b", "delta"])
    _write_csv(tables / "q2_pareto.csv", result["q2"]["rows"], list(result["q2"]["rows"][0]))

    q3_rows = []
    for condition, record in result["q3"]["conditions"].items():
        for rank, row in enumerate(record["ranking"], start=1):
            q3_rows.append({
                "condition_id": condition,
                "rank": rank,
                "system_id": row["system_id"],
                "rocch_eer": row["rocch_eer"],
                "kendall_tau_b_vs_clean": record["kendall_tau_b_vs_clean"],
                "condition_reversal_count": len(record["pairwise_rank_reversals"]),
            })
    _write_csv(tables / "q3_condition_rankings.csv", q3_rows, list(q3_rows[0]))

    cost_rows = []
    for system in SYSTEMS:
        row = result["cost"][system]
        cost_rows.append({
            "system_id": system,
            "median_ms": row["latency"]["median_ms"],
            "q1_ms": row["latency"]["q1_ms"],
            "q3_ms": row["latency"]["q3_ms"],
            "mean_ms": row["latency"]["mean_ms"],
            "repeats": row["latency"]["n"],
            "trainable_params": row["trainable_params"],
            "total_params": row["total_params"],
        })
    _write_csv(tables / "cost.csv", cost_rows, list(cost_rows[0]))

    calibration_rows = []
    for condition, condition_record in result["conditions"].items():
        for system in SYSTEMS:
            record = condition_record["systems"][system]
            if record["status"] != "complete":
                continue
            calibration_rows.append({
                "condition_id": condition,
                "system_id": system,
                "primary_clean_calibrator_cllr": record["primary_clean_calibrator"]["cllr"],
                "secondary_condition_calibrator_cllr": record["secondary_condition_calibrator"]["cllr"],
                "primary_cllr_cal": record["primary_clean_calibrator"]["cllr_cal"],
                "secondary_cllr_cal": record["secondary_condition_calibrator"]["cllr_cal"],
            })
    _write_csv(tables / "calibration_transfer.csv", calibration_rows, list(calibration_rows[0]))

    _latex_table(
        tables / "table_clean_metrics.tex",
        "lrrrrr",
        "System & ROCCH-EER & AUC & $C_{llr}$ & Brier & ECE",
        (
            f"{r['system_id']} & {_fmt(r['rocch_eer'])} & {_fmt(r['auc'])} & {_fmt(r['cllr'])} & {_fmt(r['brier'])} & {_fmt(r['ece'])} \\\\" 
            for r in clean_rows
        ),
        "Clean-condition V1 results. Lower is better for EER and calibration losses; higher is better for AUC.",
        "tab:v1-clean-results",
    )
    _latex_table(
        tables / "table_q2_pareto.tex",
        "lrrrrc",
        "System & EER & Robustness loss & $C_{llr}$ & Latency (ms) & Pareto",
        (
            f"{r['system_id']} & {_fmt(r['clean_rocch_eer'])} & {_fmt(r['mean_stress_rocch_eer_loss'])} & {_fmt(r['clean_cllr'])} & {_fmt(r['fusion_latency_median_ms'], 3)} & {'yes' if r['pareto_non_dominated'] else 'no'} \\\\" 
            for r in result["q2"]["rows"]
        ),
        "Locked four-dimensional Q2 point-estimate Pareto analysis.",
        "tab:v1-q2-pareto",
    )

    _write_figures(result, figures)
    clean_best = min(clean_rows, key=lambda row: row["rocch_eer"])
    best_dl = min((row for row in clean_rows if row["system_id"].startswith("D")), key=lambda row: row["rocch_eer"])
    output.joinpath("README.md").write_text(
        "# Locked V1 result assets\n\n"
        "Every file in this directory is generated by `scripts/generate_v1_result_assets.py` "
        "from the committed, hash-verified workflow evidence.\n\n"
        f"- Workflow run: `{V1_FINAL_RUN_ID}`, attempt `{V1_FINAL_RUN_ATTEMPT}`\n"
        f"- Workflow artifact: `{V1_FINAL_ARTIFACT_ID}`\n"
        f"- Artifact ZIP SHA-256: `{V1_FINAL_ARTIFACT_ZIP_SHA256}`\n"
        f"- Final-results lock SHA-256: `{V1_FINAL_RESULTS_LOCK_SHA256}`\n"
        "- Scope: 20 public biometric-instance identities, not independent humans and not the full NUPT-FPV archive\n"
        "- Evidence: 15 conditions, 4,000 trials per condition, 148 available system-condition manifests\n\n"
        "## Direct point-estimate findings\n\n"
        f"- Lowest clean ROCCH-EER: {clean_best['system_id']} ({clean_best['rocch_eer']:.6f}).\n"
        f"- Lowest clean ROCCH-EER among DL families: {best_dl['system_id']} ({best_dl['rocch_eer']:.6f}).\n"
        "- Q2 point frontier: "
        + ", ".join(row["system_id"] for row in result["q2"]["rows"] if row["pareto_non_dominated"])
        + ".\n"
        "- These are bounded public-instance point estimates. They are not person-population confidence claims.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/v1"))
    args = parser.parse_args()
    generate(args.evidence, args.output)
    print(json.dumps({"status": "generated", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
