"""Self-integrity lock for the final V1 scoring program."""

from __future__ import annotations

import hashlib
from pathlib import Path


V1_FINAL_SCRIPT_PATH = Path("scripts/run_v1_final_evaluation.py")
V1_FINAL_SCRIPT_SHA256 = "2a63bb8546893d4884c1cbd7a82304a827c7148a33c005483daffdd20848bfee"


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
