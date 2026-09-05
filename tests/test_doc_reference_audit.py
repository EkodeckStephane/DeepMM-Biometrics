import subprocess
import sys


def test_every_documented_doi_is_in_verified_registry():
    result = subprocess.run(
        [sys.executable, "scripts/validate_doc_references.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
