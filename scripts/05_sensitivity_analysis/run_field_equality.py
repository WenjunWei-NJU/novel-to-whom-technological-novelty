"""Blockwise FWL estimation of the fully field-interacted two-outcome model."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import duckdb
import numpy as np
import pandas as pd
from scipy import linalg

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "_shared"))
from study.analysis import subset, team_design, profile_vector, profile_support, OUTCOMES
from study.fe import TwoWayFE, fit_multioutcome, contrast, joint_wald, holm, covariance_ok
from study.io import save_json

FIELDS = ["ai", "early_share", "senior_share", "age_mean", "logq_mean", "width_mean", "first_share"]
OUT = OUTCOMES[:2]
SECTIONS = ("A", "B", "G", "H")
EXPECTED = [306562, 319660, 674250, 786788]
SUPPORT_MIN = [77, 91, 764, 234]
TOL = 1e-9  # Fraction scale: 1e-7 percentage points.
R = np.column_stack([-np.ones(3), np.eye(3)])


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(dest, formal, database):
    cfg = json.loads((formal / "analysis_settings.json").read_text())
    cfg["source_database"] = str(database.resolve())
    source_csv = formal / "tables/estimands.csv"
    reference = pd.read_csv(source_csv)
    cols = ["patent_id", "primary_cpc", "application_year", "n", *FIELDS, *OUTCOMES]
    query = ("select " + ",".join(cols) + ",1.0 weight from teams "
             "where n>=2 and n_unknown=0 and eligible_n>0 and ai is not null "
             "and application_year between 2000 and 2024 and starts_with(primary_cpc, ?) order by patent_id")
    metadata = []
    all_scores = {o: {} for o in OUT}
    results = {o: [] for o in OUT}
    fingerprints = {"estimands": file_hash(source_csv)}
    with duckdb.connect(cfg["source_database"], read_only=True) as con:
        con.execute("set threads=2")
        con.execute("set memory_limit='2GB'")
        for i, section in enumerate(SECTIONS):
            name = f"field_{section}"
            raw = con.execute(query, [section]).df()
            finite = (np.isfinite(raw[FIELDS + OUTCOMES + ["application_year", "n", "weight"]]).all(axis=1)
                      & raw.primary_cpc.notna())
            # Keep the original three-outcome sample rule while fitting two outcomes.
            data, groups = subset(raw, FIELDS + OUTCOMES)
            n_raw, n_complete = len(raw), int(finite.sum())
            del raw
            assert len(data) == EXPECTED[i]
            ids = np.sort(data.patent_id.to_numpy(dtype="<i8"))
            labels, cpc = np.unique(data.primary_cpc.to_numpy(dtype=str), return_inverse=True)
            support = profile_support(data, cfg)
            assert support["minimum_endpoint_N"] == SUPPORT_MIN[i]
            assert support["minimum_required_N"] == 20 and support["adequate_exact_support"]
            assert all(label.startswith(section) for label in labels)
            design = team_design(data, "surface")
            scales = np.asarray(design.scales)
            target = profile_vector(design.names, cfg)
            weights = data.weight.to_numpy()
            y = data[OUT].to_numpy()
            with tempfile.TemporaryDirectory(prefix="joi_field_fwl_") as cache:
                x = np.lib.format.open_memmap(Path(cache) / "X.npy", mode="w+", dtype="float64",
                                               shape=(len(data), len(scales)))
                for j, fun in enumerate(design.functions):
                    x[:, j] = fun() / scales[j]
                fit = fit_multioutcome(x, y, weights, groups, {"CPC": [cpc]}, design.names, scales,
                                         cache=cache, target_vector=target)
                rx = np.load(Path(cache) / "RX.npy", mmap_mode="r")
                # With unit weights, recover cluster influences for the target
                # from the residualized regressors and the fitted residuals.
                assert np.all(weights == 1)
                fe = TwoWayFE(groups, weights)
                gram = rx.T @ rx
                target_direction = rx @ linalg.solve(gram, target / scales, assume_a="pos")
                for j, outcome in enumerate(OUT):
                    info = fit["outcomes"][j]["covariance"]["CPC"]
                    assert info["inference_ok"]
                    beta = fit["outcomes"][j]["beta"]
                    residual = fe.project(y[:, j])[:, 0] - rx @ (beta * scales)
                    score = np.bincount(cpc, weights=target_direction * residual, minlength=len(labels))
                    result = contrast(beta, info["V"], target, info["df"])
                    correction = info["components"][0]["factor"]
                    assert np.isclose(score @ score * correction, result["variance"], rtol=1e-9, atol=1e-15)
                    official, = reference.loc[
                        (reference.model == name) & (reference.outcome == outcome)
                        & (reference.family == "FIELDS")
                    ].to_dict("records")
                    assert int(official["N"]) == len(data) and int(official["df"]) == info["df"]
                    for k in ("estimate", "SE", "CI_low", "CI_high"):
                        assert abs(result[k] - official[k]) < TOL, (name, outcome, k)
                    all_scores[outcome][section] = pd.Series(score, index=labels)
                    results[outcome].append(dict(section=section, original=official, reproduced=result))
                rx._mmap.close()
                x._mmap.close()
            old_path = formal / "models" / f"{name}.json"
            fingerprints[name] = file_hash(old_path)
            original_fit = json.loads(old_path.read_text())
            assert fit["names"] == original_fit["names"] and fit["FE_rank"] == original_fit["FE_rank"]
            for j in range(len(OUT)):
                assert np.allclose(fit["outcomes"][j]["beta"], original_fit["outcomes"][j]["beta"],
                                   rtol=1e-7, atol=TOL)
            info = fit["outcomes"][0]["covariance"]["CPC"]
            meta = dict(section=section, candidate_N=n_raw, endpoint_support=support, complete_N=n_complete, N=len(data),
                        singletons_removed=n_complete-len(data), clusters=len(labels),
                        cluster_df=info["df"], FE_rank=fit["FE_rank"], K=info["K"],
                        FE_rank_counted=info["fe_rank_counted"], CRV1=info["components"][0]["factor"],
                        patent_ID_SHA256=hashlib.sha256(ids.tobytes()).hexdigest())
            metadata.append(meta)
            save_json(dest / f"{name}_reproduction.json", dict(sample=meta, fit=fit))
            print(meta, flush=True)
    n = sum(m["N"] for m in metadata)
    k = sum(m["K"] for m in metadata)
    score_frames = {o: pd.DataFrame(all_scores[o]).fillna(0) for o in OUT}
    clusters = len(score_frames[OUT[0]])
    presence = pd.DataFrame({s: np.isin(score_frames[OUT[0]].index, v.index)
                             for s, v in all_scores[OUT[0]].items()}).astype(int)
    overlap = presence.T @ presence
    assert np.array_equal(overlap.to_numpy(), np.diag([m["clusters"] for m in metadata]))
    assert n == 2087260
    factor = clusters / (clusters - 1) * (n - 1) / (n - k)
    joint = {}
    for outcome in OUT:
        scores = score_frames[outcome]
        scores.to_csv(dest / f"{outcome}_cluster_influences.csv", index_label="primary_cpc")
        v = factor * (scores.to_numpy().T @ scores.to_numpy())
        assert np.count_nonzero(v - np.diag(v.diagonal())) == 0
        assert covariance_ok(v) and linalg.eigvalsh(v)[0] > 0
        delta = np.array([r["reproduced"]["estimate"] for r in results[outcome]])
        test = joint_wald(delta, v, R, clusters - 1)
        assert test["inference_ok"]
        ci = [contrast(delta, v, row, clusters - 1) for row in np.eye(4)]
        joint[outcome] = dict(test=test, estimate=delta, covariance=v, joint_model_pointwise=ci,
                              covariance_eigenvalues=linalg.eigvalsh(v))
    adjusted = holm([joint[o]["test"]["p_raw"] for o in OUT])
    for o, p in zip(OUT, adjusted):
        joint[o]["test"]["p_Holm"] = p
    payload = dict(N=n, G=clusters, K=k, cluster="primary_cpc", joint_CRV1_factor=factor,
                   original_eligibility_SQL=query, restrictions=R, tolerance=TOL,
                   cluster_overlap=overlap.to_dict(),
                   sample=metadata, reproduction=results, joint=joint,
                   holm_family="EXPLORATORY_FIELD_EQUALITY", holm_m=2,
                   original_settings=cfg, original_file_SHA256=fingerprints)
    save_json(dest / "field_equality_results.json", payload)
    for outcome in OUT:
        print(outcome, joint[outcome]["test"], flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--sensitivity-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.database, args.sensitivity_results / "analysis_settings.json",
                 args.sensitivity_results / "tables/estimands.csv"):
        if not path.is_file():
            parser.error(f"Required input file does not exist: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    run(args.output, args.sensitivity_results, args.database)
