from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "_shared"))
from study.fe import contrast, holm, joint_wald

parser = argparse.ArgumentParser(description="Compute the six stage pairwise contrasts.")
parser.add_argument("--stage-results", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
arguments = parser.parse_args()
for path in (
    arguments.stage_results / "models/personal_stage.json",
    arguments.stage_results / "tables/estimands.csv",
):
    if not path.is_file():
        parser.error(f"Required input file does not exist: {path}")
stage = arguments.stage_results.resolve()
output = arguments.output.resolve()
output.mkdir(parents=True, exist_ok=False)
model = json.loads((stage / "models/personal_stage.json").read_text())
assert model["N"] == 12_908_659 and model["synthetic"] is False
fit = model["outcomes"][0]
beta = np.array(fit["beta"])
basis = dict(zip(model["names"], np.eye(len(beta))))
vectors = {
    "Early": basis["E_AI"] + basis["E_AI_stage_E"],
    "Mid": basis["E_AI"],
    "Senior": basis["E_AI"] + basis["E_AI_stage_S"],
}
official = pd.read_csv(stage / "tables/estimands.csv")
fields = ["estimate", "SE", "CI_low", "CI_high", "p_raw", "df"]
rows = []
for cluster, label in [
    ("INV_PAT", "Inventor + patent"),
    ("INV_CPC", "Inventor + primary CPC subclass"),
]:
    info = fit["covariance"][cluster]
    covariance = np.array(info["V"])
    assert covariance.shape == (23, 23)
    np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=1e-14)
    args = (info["df"], info["inference_ok"])
    published = official[official.covariance.eq(cluster)].set_index("outcome")
    for name, code in zip(vectors, ["E", "M", "S"]):
        result = contrast(beta, covariance, vectors[name], *args)
        assert result["inference_ok"]
        assert published.loc[code, "N"] == model["N"]
        np.testing.assert_allclose(
            [result[key] for key in fields],
            published.loc[code, fields].to_numpy(float), rtol=1e-11, atol=0,
        )
    restrictions = [basis["E_AI_stage_E"], basis["E_AI_stage_S"]]
    joint = joint_wald(beta, covariance, restrictions, *args)
    for key in ["F", "df_numerator", "df_denominator", "p_raw"]:
        np.testing.assert_allclose(joint[key], published.loc["all", key], rtol=1e-11)
    for high, low in [("Mid", "Early"), ("Senior", "Early"), ("Senior", "Mid")]:
        result = contrast(beta, covariance, vectors[high] - vectors[low], *args)
        assert result["inference_ok"]
        rows.append({
            "comparison": f"{high} − {low}", "clustering": label, "df": result["df"],
            **{f"{key}_pp": result[key] * 100 for key in fields[:4]},
            "p_raw": result["p_raw"],
        })
table = pd.DataFrame(rows)
table["p_Holm_6"] = holm(table.p_raw)
table.to_csv(output / "PAIRWISE_RESULTS.csv", index=False)
header = "| 比较 | 双向聚类 | df | 差值 (pp) | SE (pp) | 95% CI (pp) | 原始 p | Holm p（6项） |\n|---|---|---:|---:|---:|---:|---:|---:|"
lines = [header]
for row in table.itertuples():
    lines.append(
        f"| {row.comparison} | {row.clustering} | {row.df:,} | {row.estimate_pp:.4f} | "
        f"{row.SE_pp:.4f} | [{row.CI_low_pp:.4f}, {row.CI_high_pp:.4f}] | "
        f"{row.p_raw:.6g} | {row.p_Holm_6:.6g} |"
    )
(output / "PAIRWISE_RESULTS.md").write_text("# 职业阶段 E × AI 斜率差逐对检验\n\n" + "\n".join(lines) + "\n\n95% CI为逐点区间。六项双侧检验共同构成独立的探索性Holm家族。\n")
print("\n".join(lines))
