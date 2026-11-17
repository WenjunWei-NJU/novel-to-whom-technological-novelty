from __future__ import annotations

import glob
import json
import os
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def ident(value):
    return '"' + str(value).replace('"', '""') + '"'


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    root = Path(os.path.expandvars(os.path.expanduser(config.get("data_root", "."))))
    if not root.is_absolute():
        root = path.parent / root
    config["data_root"] = str(root.resolve())
    config.setdefault("snapshot", "2025-12-30")
    config.setdefault("cohort_start", 2000)
    config.setdefault("cohort_end", 2024)
    config.setdefault("primary_end", 2024)
    config.setdefault("early_years", 5)
    config.setdefault("senior_years", 16)
    config.setdefault("ai_threshold", .5)
    config.setdefault("windows", [5, 10])
    config.setdefault("threads", 4)
    config.setdefault("memory_limit", "4GB")
    config.setdefault("plot_sample", 12000)
    config.setdefault("min_surface_cell", 20)
    config.setdefault("figure_dpi", 450)
    config.setdefault("figure_width_mm", 183)
    config.setdefault("robustness", True)
    for name in ["cohort_start", "cohort_end", "primary_end", "threads", "plot_sample", "min_surface_cell", "figure_dpi"]:
        value=config[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or int(value)!=value:
            raise ValueError(f"{name} must be a finite whole number")
        if value<1:
            raise ValueError(f"{name} must be positive")
        config[name]=int(value)
    for name in ["early_years","senior_years","ai_threshold","figure_width_mm"]:
        value=config[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite and numeric")
    if config["figure_width_mm"]<=0:
        raise ValueError("figure_width_mm must be positive")
    try:
        config["snapshot"]=date.fromisoformat(config["snapshot"]).isoformat()
    except (TypeError,ValueError)as exc:
        raise ValueError("snapshot must be a calendar date in YYYY-MM-DD format")from exc
    if not 0 < config["early_years"] < config["senior_years"]:
        raise ValueError("Career-stage cutoffs must satisfy 0 < early < senior")
    if not config["cohort_start"] <= config["primary_end"] <= config["cohort_end"]:
        raise ValueError("cohort_start <= primary_end <= cohort_end is required")
    if not 0 <= config["ai_threshold"] <= 1:
        raise ValueError("AI threshold must lie in [0,1]")
    if not isinstance(config["windows"],list) or any(isinstance(w,bool) or not isinstance(w,(int,float)) or not np.isfinite(w) or int(w) != w or w <= 0 for w in config["windows"]):
        raise ValueError("History windows must be positive whole calendar years")
    config["windows"]=[int(w)for w in config["windows"]]
    if len(set(config["windows"]))!=len(config["windows"]):
        raise ValueError("History windows must be unique")
    return config


def connect(out, config, read_only=False):
    out = Path(out)
    con = duckdb.connect(str(out / "study.duckdb"), read_only=read_only)
    if not read_only:
        (out / "temporary").mkdir(exist_ok=True)
        con.execute(f"SET threads={int(config['threads'])}")
        con.execute(f"SET memory_limit={quote(config['memory_limit'])}")
        con.execute(f"SET temp_directory={quote(out / 'temporary')}")
        con.execute("SET preserve_insertion_order=false")
    return con


def source(con, name, config, default_columns, required=True):
    item = config.get("inputs", {}).get(name)
    if not item:
        if required:
            raise ValueError(f"inputs.{name} is required")
        return None
    item = dict(path=item) if isinstance(item, str) else item
    if not isinstance(item,dict) or not isinstance(item.get("path"),str):
        raise ValueError(f"inputs.{name} must specify a file path")
    path = Path(os.path.expandvars(os.path.expanduser(item["path"])))
    if not path.is_absolute():
        path = Path(config["data_root"]) / path
    paths = [str(path)] if path.is_file() else sorted(glob.glob(str(path)))
    if not paths or any(not Path(p).is_file() for p in paths):
        raise FileNotFoundError(str(path))
    suffix = Path(paths[0]).suffix.lower()
    literal = "[" + ",".join(quote(p) for p in paths) + "]"
    if suffix == ".parquet":
        query = f"SELECT * FROM read_parquet({literal}, union_by_name=true)"
    else:
        sep = item.get("separator", "\t" if any(".tsv" in p for p in paths) else ",")
        query = f"SELECT * FROM read_csv_auto({literal},delim={quote(sep)},header=true,all_varchar=true,union_by_name=true)"
    con.execute(f"CREATE OR REPLACE VIEW raw_{name} AS {query}")
    available = {row[0] for row in con.execute(f"DESCRIBE raw_{name}").fetchall()}
    overrides=item.get("columns",{})
    if not isinstance(overrides,dict) or set(overrides)-set(default_columns):
        raise ValueError(f"Unknown canonical column names for inputs.{name}")
    columns = dict(default_columns, **overrides)
    mapped=[value for value in columns.values()if value is not None]
    if any(not isinstance(value,str)or not value for value in mapped)or len(set(mapped))!=len(mapped):
        raise ValueError(f"Column mappings for inputs.{name} must use distinct nonempty source names")
    return dict(view=f"raw_{name}", available=available, columns=columns)


def field(src, key, cast=None, optional=False, default="NULL"):
    value = src["columns"].get(key)
    if value not in src["available"]:
        if optional:
            return default
        raise ValueError(f"Missing source column for {key}: {value}")
    expression = ident(value)
    return f"TRY_CAST({expression} AS {cast})" if cast else expression


def require_unique(con, table, keys):
    expression = ",".join(ident(k) for k in keys)
    missing = " OR ".join(f"{ident(k)} IS NULL OR trim(CAST({ident(k)} AS VARCHAR))=''" for k in keys)
    if con.execute(f"SELECT count(*) FROM {table} WHERE {missing}").fetchone()[0]:
        raise ValueError(f"Missing keys in {table}")
    if con.execute(f"SELECT count(*) FROM (SELECT {expression} FROM {table} GROUP BY {expression} HAVING count(*)>1)").fetchone()[0]:
        raise ValueError(f"Duplicate keys in {table}: {keys}")


def export(con, query, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = "FORMAT PARQUET, COMPRESSION ZSTD" if path.suffix == ".parquet" else "FORMAT CSV, HEADER true"
    con.execute(f"COPY ({query}) TO {quote(path)} ({options})")


def frame(con, query):
    return con.execute(query).df()


def csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
