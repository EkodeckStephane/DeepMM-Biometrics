"""Self-integrity lock for the final V1 scoring program."""

from __future__ import annotations

import hashlib
from pathlib import Path


V1_FINAL_SCRIPT_PATH = Path("scripts/run_v1_final_evaluation.py")
V1_FINAL_SCRIPT_SHA256 = "6ab5b5e90a8027a56d467bad3cf792d74c7463e5b55c49f2ac913a93e30a88bc"


def assert_v1_final_script_lock(path: Path | None = None) -> None:
    source = (
        Path(__file__).resolve().parents[3] / V1_FINAL_SCRIPT_PATH
        if path is None
        else Path(path)
    )
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != V1_FINAL_SCRIPT_SHA256:
        raise RuntimeError(
            f"V1 final runner changed: expected {V1_FINAL_SCRIPT_SHA256}, got {actual}"
        )
