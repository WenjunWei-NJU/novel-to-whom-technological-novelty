"""Fast input-schema checks before building the large history tables."""
from __future__ import annotations

import duckdb

from .io import field, source


SOURCES = {
    "patents": (dict(patent_id="patent_id", filing_date="filing_date", grant_date="patent_date", patent_type="patent_type"), ["patent_id", "grant_date"]),
    "applications": (dict(patent_id="patent_id", filing_date="filing_date"), ["patent_id", "filing_date"]),
    "inventors": (dict(patent_id="patent_id", inventor_id="inventor_id"), ["patent_id", "inventor_id"]),
    "cpc": (dict(patent_id="patent_id", cpc="cpc_subclass", sequence="cpc_sequence"), ["patent_id", "cpc"]),
    "ai": (dict(patent_id="patent_id", score="aipat_ai_score"), ["patent_id", "score"]),
    "locations": (dict(patent_id="patent_id", latitude="latitude", longitude="longitude"), ["patent_id", "latitude", "longitude"]),
    "citations": (dict(citing="patent_id", cited="citation_patent_id"), ["citing", "cited"]),
    "identity_issues": (dict(inventor_id="inventor_id"), ["inventor_id"]),
}


def check_inputs(config):
    """Read schemas only; do not create results or compute history/outcome tables.

    Row-level date, score, key and history checks remain part of the full build.
    """
    checks, problems = [], []
    with duckdb.connect() as con:
        con.execute("SET threads=1; SET memory_limit='256MB'")
        for name, (columns, required) in SOURCES.items():
            must_exist = name in {"patents", "inventors", "cpc", "ai"}
            try:
                src = source(con, name, config, columns, required=must_exist)
                if src is None:
                    continue
                required = list(required)
                if name == "patents":
                    if not config.get("inputs", {}).get("applications"):
                        required.append("filing_date")
                    if not config.get("utility_only_input", False):
                        required.append("patent_type")
                for key in required:
                    field(src, key)
                checks.append(dict(source=name, status="ready", fields=required))
            except (OSError, ValueError, duckdb.Error) as error:
                problems.append(dict(source=name, error=str(error)))
    return dict(status="READY" if not problems else "INPUTS_INCOMPLETE",
                scope="file paths and schemas only; no row-level or estimation certification",
                sources=checks, problems=problems)
