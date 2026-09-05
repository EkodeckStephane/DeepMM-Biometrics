import subprocess
import sys


def test_versioned_literature_registry_and_bibtex_are_synchronized():
    result = subprocess.run(
        [sys.executable, "scripts/validate_literature.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "BibTeX synchronized" in result.stdout
