#!/usr/bin/env python3
"""Validate the versioned SOTA registry against the BibTeX seed bibliography.

This is an *offline consistency* validator. It does not query DOI registries and
therefore cannot prove that a reference exists. Existence/claim support is checked
manually against authoritative sources before an entry receives
``metadata_status=verified``; this script prevents those verified fields from
silently diverging across repository artifacts.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
BIB_ENTRY_RE = re.compile(r"(?ms)^@\w+\{\s*([^,\s]+)\s*,(.*?)(?=^@\w+\{|\Z)")
FIELD_RE_TEMPLATE = r"(?ims)^\s*{field}\s*=\s*[{{\"](.*?)[}}\"]\s*,?\s*$"

REQUIRED_COLUMNS = {
    "key",
    "year",
    "title",
    "venue",
    "doi",
    "category",
    "role",
    "code_status",
    "metadata_status",
}
ALLOWED_CODE_STATUS = {
    "official_public_code",
    # Publisher/article explicitly reports a code repository, but the repository
    # itself has not yet passed the DeepMM reproducibility/code-provenance audit.
    "publisher_reports_github",
    "public_repository_data_request",
    "not_located",
    "not_applicable",
    "not_a_code_baseline",
}


def _field(body: str, name: str) -> str | None:
    match = re.search(FIELD_RE_TEMPLATE.format(field=re.escape(name)), body)
    return match.group(1).strip() if match else None


def _normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).casefold()
    text = text.replace("---", "-").replace("--", "-")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def read_registry(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("registry has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"registry missing columns: {sorted(missing)}")
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def read_bib(path: Path) -> dict[str, dict[str, str | None]]:
    text = path.read_text(encoding="utf-8")
    entries: dict[str, dict[str, str | None]] = {}
    for match in BIB_ENTRY_RE.finditer(text):
        key = match.group(1).strip()
        body = match.group(2)
        if key in entries:
            raise ValueError(f"duplicate BibTeX key: {key}")
        entries[key] = {
            "doi": _field(body, "doi"),
            "year": _field(body, "year"),
            "title": _field(body, "title"),
        }
    if not entries:
        raise ValueError("no BibTeX entries found")
    return entries


def validate(registry: list[dict[str, str]], bib: dict[str, dict[str, str | None]]) -> list[str]:
    errors: list[str] = []
    seen_keys: set[str] = set()
    seen_dois: set[str] = set()

    for line_no, row in enumerate(registry, start=2):
        key = row["key"]
        doi = row["doi"].lower()
        if not key:
            errors.append(f"registry line {line_no}: empty key")
            continue
        if key in seen_keys:
            errors.append(f"registry line {line_no}: duplicate key {key}")
        seen_keys.add(key)

        try:
            year = int(row["year"])
            if not 1900 <= year <= 2100:
                raise ValueError
        except ValueError:
            errors.append(f"{key}: invalid year {row['year']!r}")

        if not row["title"] or not row["venue"] or not row["role"]:
            errors.append(f"{key}: title, venue and role must be non-empty")
        if not DOI_RE.match(row["doi"]):
            errors.append(f"{key}: malformed DOI {row['doi']!r}")
        if doi in seen_dois:
            errors.append(f"{key}: duplicate DOI {row['doi']}")
        seen_dois.add(doi)

        if row["metadata_status"] != "verified":
            errors.append(f"{key}: metadata_status must be 'verified' before entering this registry")
        if row["code_status"] not in ALLOWED_CODE_STATUS:
            errors.append(f"{key}: unknown code_status {row['code_status']!r}")

        entry = bib.get(key)
        if entry is None:
            errors.append(f"{key}: missing from literature/references.bib")
            continue
        bib_doi = (entry["doi"] or "").lower()
        if bib_doi != doi:
            errors.append(f"{key}: DOI mismatch registry={doi!r}, bib={bib_doi!r}")
        if entry["year"] != row["year"]:
            errors.append(f"{key}: year mismatch registry={row['year']!r}, bib={entry['year']!r}")
        if entry["title"] is None:
            errors.append(f"{key}: BibTeX title missing")
        elif _normalize_title(entry["title"]) != _normalize_title(row["title"]):
            errors.append(f"{key}: title mismatch between registry and BibTeX")

    extra = sorted(set(bib) - seen_keys)
    if extra:
        errors.append(f"BibTeX contains unregistered entries: {extra}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("literature/sota_registry.csv"))
    parser.add_argument("--bib", type=Path, default=Path("literature/references.bib"))
    args = parser.parse_args(argv)

    try:
        registry = read_registry(args.registry)
        bib = read_bib(args.bib)
        errors = validate(registry, bib)
    except (OSError, ValueError) as exc:
        print(f"literature validation failed: {exc}", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Literature registry OK: {len(registry)} verified entries, BibTeX synchronized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
