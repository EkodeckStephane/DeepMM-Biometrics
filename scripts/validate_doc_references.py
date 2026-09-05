#!/usr/bin/env python3
"""Ensure DOI-bearing scientific references in project docs exist in SOTA registry.

This does not prove that a cited paper supports a sentence; it prevents an easier
failure mode where a DOI is mentioned in protocol/SOTA documentation but is absent
from the versioned verified-reference registry.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s`<>\"\]\[]+", re.IGNORECASE)
TRAILING = ".,;:)]}"


def normalize_doi(value: str) -> str:
    text = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.rstrip(TRAILING)


def registry_dois(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    dois = {normalize_doi(row.get("doi", "")) for row in rows if row.get("doi")}
    if not dois:
        raise ValueError("registry contains no DOI values")
    return dois


def scan_markdown(paths: list[Path]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for root in paths:
        candidates = [root] if root.is_file() else sorted(root.rglob("*.md"))
        for path in candidates:
            if not path.exists() or path.suffix.lower() != ".md":
                continue
            text = path.read_text(encoding="utf-8")
            for match in DOI_RE.finditer(text):
                doi = normalize_doi(match.group(0))
                found.setdefault(doi, set()).add(path.as_posix())
    return found


def validate(registry: Path, roots: list[Path]) -> list[str]:
    known = registry_dois(registry)
    mentioned = scan_markdown(roots)
    errors: list[str] = []
    for doi, paths in sorted(mentioned.items()):
        if doi not in known:
            errors.append(f"unregistered DOI {doi} mentioned in {', '.join(sorted(paths))}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("literature/sota_registry.csv"))
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=[Path("docs"), Path("README.md")],
    )
    args = parser.parse_args(argv)

    try:
        errors = validate(args.registry, list(args.roots))
    except (OSError, ValueError) as exc:
        print(f"document-reference validation failed: {exc}", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Document DOI audit OK: every DOI mention is registered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
