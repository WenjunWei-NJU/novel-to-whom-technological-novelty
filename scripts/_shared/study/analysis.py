"""Multiscale estimands, common samples, composition surfaces and robustness."""
from __future__ import annotations

import gc
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .fe import IdentificationError, contrast, fit_multioutcome, holm, joint_wald
from .io import csv, frame, save_json


OUTCOMES = ["relative_y", "team_y", "pair_y"]
LABELS = dict(relative_y="Mean personal novelty", team_y="Team novelty", pair_y="Corpus-first pairs")
COMPOSITION = ["early_share", "senior_share", "early_sq", "senior_sq", "early_senior"]


def subset(d, fields):
    if "weight" not in d or not np.isfinite(d.weight.to_numpy(dtype=float)).all() or (d.weight<=0).any():
        raise ValueError("Initial analysis weights must be finite and strictly positive")
    keys=["patent_id"]+(["inventor_id"]if "inventor_id"in d else[])
    if d[keys].isna().any().any() or d.duplicated(keys).any():
        raise ValueError("Analysis observations must have unique nonmissing keys")
    numeric = list(dict.fromkeys(fields + ["application_year", "n", "weight"]))
    mask = np.isfinite(d[numeric].to_numpy(dtype=float)).all(axis=1) & d.primary_cpc.notna().to_numpy()
    data = d.loc[mask].copy()
    if not len(data):
        raise IdentificationError("No eligible common observations")
    cy = pd.factorize(pd.MultiIndex.from_frame(data[["primary_cpc", "application_year"]]), sort=True)[0]
    size = pd.factorize(data.n, sort=True)[0]
    keep = np.ones(len(data), bool)
    while keep.any():
        a = np.bincount(cy[keep], minlength=cy.max() + 1)
        b = np.bincount(size[keep], minlength=size.max() + 1)
        remove = keep & ((a[cy] == 1) | (b[size] == 1))
        if not remove.any():
            break
        keep[remove] = False
    data = data.loc[keep].reset_index(drop=True)
    if len(data) < 40:
        raise IdentificationError("Too few observations after FE singleton removal")
    groups = [pd.factorize(pd.MultiIndex.from_frame(data[["primary_cpc", "application_year"]]), sort=True)[0],
              pd.factorize(data.n, sort=True)[0]]
    return data, groups


class Design:
    def __init__(self):
        self.names, self.scales, self.functions = [], [], []

    def add(self, name, function, scale=1):
        self.names.append(name)
        self.scales.append(scale)
        self.functions.append(function)

    def values(self, name, values, scale=1):
        self.add(name, lambda v=values: v, scale)

    def interacting(self, name, function, ai, scale=1):
        self.add(name, function, scale)
        self.add(name + "_AI", lambda f=function: f() * ai, scale)


def team_design(data, kind="surface"):
    d = Design()
    ai = data.ai.to_numpy(dtype=float)
    early, senior = data.early_share.to_numpy(), data.senior_share.to_numpy()
    d.values("AI", ai)
    if kind == "distance":
        distance = data.mean_age_distance.to_numpy() / 10
        d.interacting("career_distance_10y", lambda: distance, ai)
    else:
        d.interacting("early_share", lambda: early, ai)
        d.interacting("senior_share", lambda: senior, ai)
        if kind in {"surface", "denominator"}:
            d.interacting("early_sq", lambda: early ** 2, ai)
            d.interacting("senior_sq", lambda: senior ** 2, ai)
            d.interacting("early_senior", lambda: early * senior, ai)
    age, logq = data.age_mean.to_numpy() - 20, data.logq_mean.to_numpy()
    width, first = data.width_mean.to_numpy(), data.first_share.to_numpy()
    for name, fn, scale in [
        ("mean_age", lambda: age, 10), ("mean_age_sq", lambda: age ** 2, 100),
        ("mean_logq", lambda: logq, 1), ("mean_width", lambda: width, 10),
        ("FIRST", lambda: first, 1)]:
        d.interacting(name, fn, ai, scale)
    if kind == "denominator":
        d.interacting("log_pair_n", lambda: np.log(data.pair_n.to_numpy()), ai)
    return d


def personal_design(data, kind="full"):
    d = Design()
    ai, e = data.ai.to_numpy(dtype=float), data.early_peers.to_numpy()
    age, logq = data.age.to_numpy() - 20, np.log1p(data.prior_patents.to_numpy())
    d.values("AI", ai)
    d.interacting("E", lambda: e, ai)
    for name, fn, scale in [("age", lambda: age, 10), ("age_sq", lambda: age ** 2, 100),
                             ("logq", lambda: logq, 1), ("width", lambda: data.history_width.to_numpy(), 10)]:
        d.interacting(name, fn, ai, scale)
    for name, column in [("FIRSTpeer", "first_peers"), ("SENIORpeer", "senior_peers")]:
        values = data[column].to_numpy()
        if kind == "baseline":
            d.values(name, values)
        else:
            d.interacting(name, lambda v=values: v, ai)
    if kind == "stage":
        for label in ["E", "S"]:
            indicator = (data.stage == label).to_numpy(dtype=float)
            d.values("stage_" + label, indicator)
            d.values("AI_stage_" + label, ai * indicator)
            d.values("E_stage_" + label, e * indicator)
            d.values("E_AI_stage_" + label, e * ai * indicator)
    if kind == "opportunity":
        d.interacting("log_opportunity", lambda: np.log1p(data.opportunity_n.to_numpy()), ai)
    return d


def profile_vector(names, cfg, ai=True):
    spec = cfg.get("profile", dict(n=5, start=[1, 1], end=[2, 2]))
    n = float(spec["n"])
    counts=np.asarray([spec["start"],spec["end"]],dtype=float)
    first=float(spec.get("first_count",0))
    if (not np.isfinite(n) or n<2 or n!=np.floor(n) or counts.shape!=(2,2)
            or not np.isfinite(counts).all() or np.any(counts<0)
            or np.any(counts!=np.floor(counts)) or np.any(counts.sum(axis=1)>n)
            or not np.isfinite(first) or first<0 or first!=np.floor(first)
            or first>counts[:,0].min()):
        raise ValueError("Invalid team composition profile")
    a,b=counts/n
    delta = np.array([b[0]-a[0], b[1]-a[1], b[0]**2-a[0]**2,
                      b[1]**2-a[1]**2, b[0]*b[1]-a[0]*a[1]])
    vector = np.zeros(len(names))
    for key, value in zip(COMPOSITION, delta):
        name = key + "_AI" if ai else key
        if name in names:
            vector[names.index(name)] = value
    return vector


def profile_support(data,cfg):
    """Exact composition support; it is distinct from structural feasibility."""
    spec=cfg.get("profile",dict(n=5,start=[1,1],end=[2,2]))
    profile_vector([],cfg)  # validate the integer profile even for linear models
    n,first=int(spec["n"]),int(spec.get("first_count",0))
    binary=set(np.unique(data.ai.to_numpy(dtype=float)))<={0.,1.}
    ai_groups=data.ai.to_numpy(dtype=int)if binary else(data.ai_score>=cfg.get("ai_threshold",.5)).to_numpy(dtype=int)
    early_share=data.early_share.to_numpy()if "early_share"in data else(data.n_early/data.n).to_numpy()
    senior_share=data.senior_share.to_numpy()if "senior_share"in data else(data.n_senior/data.n).to_numpy()
    rows=[]
    for endpoint in["start","end"]:
        early,senior=spec[endpoint]
        mask=(data.n==n).to_numpy()
        mask&=np.isclose(early_share,early/n,atol=1e-12,rtol=0)
        mask&=np.isclose(senior_share,senior/n,atol=1e-12,rtol=0)
        mask&=np.isclose(data.first_share.to_numpy(),first/n,atol=1e-12,rtol=0)
        for ai in[0,1]:
            rows.append(dict(endpoint=endpoint,AI=ai,N=int(np.sum(mask&(ai_groups==ai)))))
    minimum=min(row["N"]for row in rows)
    threshold=int(cfg.get("min_profile_cell",cfg.get("min_surface_cell",20)))
    return dict(n=n,first_count=first,start=list(spec["start"]),end=list(spec["end"]),
                moderator_type="binary"if binary else"continuous",
                minimum_endpoint_N=minimum,minimum_required_N=threshold,
                adequate_exact_support=minimum>=threshold,cells=rows,
                support_group_definition="model AI groups"if binary else"primary-threshold groups; not exact continuous-score endpoints",
                moderator_min=float(data.ai.min()),moderator_max=float(data.ai.max()))


def period_ranges(cfg):
    low,end,extended=int(cfg["cohort_start"]),int(cfg["primary_end"]),int(cfg["cohort_end"])
    periods=[]
    for a,b in[(2000,2009),(2010,2014),(2015,2019),(2020,end)]:
        candidate=(max(low,a),min(end,b))
        if candidate[0]<=candidate[1]and candidate not in periods:periods.append(candidate)
    if(low,extended)not in periods:periods.append((low,extended))
    return periods


def planned_tests(cfg):
    """Every planned test appears in the output, including unavailable tests."""
    rows=[]
    def add(family,model,outcome,label,covariance="CPC"):
        rows.append(dict(family=family,model=model,outcome=outcome,contrast=label,covariance=covariance))
    for outcome in OUTCOMES:
        add("MAIN","main_surface",outcome,"profile_AI_difference")
        add("MAIN","main_surface",outcome,"AI_composition_joint")
    add("MAIN","main_surface","relative_minus_team","profile_AI_difference")
    for outcome in OUTCOMES[:2]:add("ELEMENT_SCALES","element_scales",outcome,"profile_AI_difference")
    for outcome in OUTCOMES:
        add("CONTINUOUS_DISTANCE","career_distance",outcome,"10y_AI_difference")
        add("DENOMINATOR","pair_denominator",outcome,"profile_AI_difference")
        add("FUNCTIONAL_FORM","linear_composition",outcome,"profile_AI_difference")
    for cov in["INV_PAT","INV_CPC"]:
        for label in["E_AI","peer_AI_restrictions"]:add("PERSONAL","personal_full","y",label,cov)
        for stage in["E","M","S"]:add("CAREER_STAGE","personal_stage",stage,"E_AI",cov)
        add("CAREER_STAGE","personal_stage","all","stage_equality",cov)
        for outcome in["y","A","B","C","D"]:add("SOURCE_COMPOSITION","source_composition",outcome,"E_AI",cov)
        add("OPPORTUNITY","opportunity_utilization","utilization","E_AI",cov)
    if cfg.get("robustness",True):
        for early in[3,5,7]:
            for senior in[15,16,20]:
                for outcome in OUTCOMES:add("THRESHOLDS",f"threshold_E{early}_S{senior}",outcome,"profile_AI_difference")
        for a,b in period_ranges(cfg):
            for outcome in OUTCOMES:add("PERIODS",f"period_{a}_{b}",outcome,"profile_AI_difference")
        for w in cfg.get("windows",[5,10]):
            for ref in["full",str(w)]:
                for outcome in OUTCOMES:add("HISTORY_WINDOWS",f"history_{w}_{ref}",outcome,"profile_AI_difference")
        for name in["strict_history","left_boundary_excluded"]:
            for outcome in OUTCOMES:add("HISTORY_COMPLETENESS",name,outcome,"profile_AI_difference")
        for name in["p30","p70","continuous"]:
            for outcome in OUTCOMES:add("AI_MEASUREMENT","ai_"+name,outcome,"profile_AI_difference")
        for section in"ABCDEFGHY":
            for outcome in OUTCOMES:add("FIELDS","field_"+section,outcome,"profile_AI_difference")
    if cfg.get("inputs",{}).get("citations"):
        for outcome in OUTCOMES:add("IMPACT_VALIDATION","impact_validation",outcome,"log_5y_citations")
    return rows


class Analysis:
    def __init__(self, con, cfg, out):
        self.con, self.cfg, self.out = con, cfg, Path(out)
        self.estimands, self.samples, self.status = [], [], []
        self.coefficients=[]
        self.profile_rows=[]
        self.out.joinpath("models").mkdir(exist_ok=True)
        self.cache = self.out / "temporary/model_cache"

    def fit(self, name, data, fields, outcomes, kind, personal=False, covs=None):
        try:
            data, groups = subset(data, fields + outcomes)
            design = personal_design(data, kind) if personal else team_design(data, kind)
            folder = self.cache / name
            folder.mkdir(parents=True, exist_ok=True)
            x = np.lib.format.open_memmap(folder / "X.npy", mode="w+", dtype="float64", shape=(len(data), len(design.names)))
            for j, fn in enumerate(design.functions):
                x[:, j] = fn() / design.scales[j]
            x.flush()
            cpc = pd.factorize(data.primary_cpc, sort=True)[0]
            if personal:
                inv = pd.factorize(data.inventor_id, sort=True)[0]
                pat = pd.factorize(data.patent_id, sort=True)[0]
                clustering = {"INV_PAT": [inv, pat], "INV_CPC": [inv, cpc]}
                if covs:
                    clustering = {k: clustering[k] for k in covs}
                target = np.eye(len(design.names))[design.names.index("E_AI")]
            else:
                clustering = {"CPC": [cpc]}
                target = (np.eye(len(design.names))[design.names.index("career_distance_10y_AI")]
                          if kind == "distance" else profile_vector(design.names, self.cfg))
            print(f"{name}: {len(data):,} observations, {len(design.names)} columns", flush=True)
            fit = fit_multioutcome(x, data[outcomes].to_numpy(dtype=float), data.weight.to_numpy(), groups,
                                   clustering, design.names, design.scales, cache=folder,
                                   target="E_AI" if personal else "career_distance_10y_AI" if kind == "distance" else "early_share_AI",
                                   target_vector=target)
            fit.update(model=name, outcome_names=outcomes, synthetic=bool(self.cfg.get("synthetic")), kind=kind)
            if not personal and kind!="distance":
                support=profile_support(data,self.cfg)
                fit["profile_support"]=support
                for cell in support["cells"]:
                    self.profile_rows.append(dict(model=name,n=support["n"],first_count=support["first_count"],
                        minimum_required_N=support["minimum_required_N"],adequate_exact_support=support["adequate_exact_support"],
                        moderator_type=support["moderator_type"],support_kind="exact_n_FIRST_count",**cell))
            self.store_coefficients(fit)
            self.samples.append(dict(model=name, N=len(data), patents=int(data.patent_id.nunique()),
                                     inventors=int(data.inventor_id.nunique()) if personal else None,
                                     AI_N=int((data.ai_score>=self.cfg['ai_threshold']).sum()), FE_rank=fit["FE_rank"], weights_renormalized=False))
            self.status.append(dict(model=name, status="estimated", N=len(data)))
            save_json(self.out / "models" / f"{name}.json", fit)
            x._mmap.close()
            del x
            gc.collect()
            shutil.rmtree(folder)
            return fit, data
        except (IdentificationError, ArithmeticError, ValueError) as error:
            message=str(error)
            error.__traceback__=None
            if "x"in locals():
                x._mmap.close();del x
            gc.collect()
            if "folder"in locals()and folder.exists():shutil.rmtree(folder)
            self.status.append(dict(model=name, status="not_estimable", reason=message))
            save_json(self.out / "models" / f"{name}.json", dict(model=name, status="not_estimable", reason=message))
            return None, None

    def store_coefficients(self,fit):
        for outcome,result in zip(fit["outcome_names"],fit["outcomes"]):
            for covariance,info in result["covariance"].items():
                for j,name in enumerate(fit["names"]):
                    value=contrast(result["beta"],info["V"],np.eye(len(fit["names"]))[j],info["df"],info["inference_ok"])
                    self.coefficients.append(dict(model=fit["model"],outcome=outcome,covariance=covariance,variable=name,N=fit["N"],K=info["K"],data_kind="SYNTHETIC"if self.cfg.get("synthetic")else"OBSERVED",**value))

    def add(self, family, model, outcome, label, result, **extra):
        extra.setdefault("covariance","CPC")
        self.estimands.append(dict(family=family, model=model, outcome=outcome, contrast=label,
                                  **result, **extra))

    def summarize_team(self, fit, family, cfg=None):
        if fit is None:
            return
        cfg = cfg or self.cfg
        names = fit["names"]
        vector = profile_vector(names, cfg)
        support=fit.get("profile_support",{})
        support_fields=dict(profile_support_ok=support.get("adequate_exact_support"),
                            profile_minimum_N=support.get("minimum_endpoint_N"))
        for outcome, result in zip(fit["outcome_names"], fit["outcomes"]):
            info = result["covariance"]["CPC"]
            self.add(family, fit["model"], outcome, "profile_AI_difference",
                     contrast(result["beta"], info["V"], vector, info["df"], info["inference_ok"]), N=fit["N"],**support_fields)
            if family == "MAIN":
                r = np.vstack([np.eye(len(names))[names.index(x + "_AI")] for x in COMPOSITION])
                self.add(family, fit["model"], outcome, "AI_composition_joint",
                         joint_wald(result["beta"], info["V"], r, info["df"], info["inference_ok"]), N=fit["N"])
        if family == "MAIN":
            if not np.allclose(vector,fit["target_vector"],rtol=0,atol=1e-12):
                raise ValueError("Cross-outcome covariance was computed for a different profile contrast")
            beta = np.array([r["beta"] @ vector for r in fit["outcomes"]])
            v = fit["target_covariances"]["CPC"]
            inference = all(fit["outcomes"][i]["covariance"]["CPC"]["inference_ok"] for i in [0, 1])
            self.add(family, fit["model"], "relative_minus_team", "profile_AI_difference",
                     contrast(beta, v, [1, -1, 0], fit["cluster_df"]["CPC"], inference), N=fit["N"],**support_fields)

    def run(self):
        end = int(self.cfg["primary_end"])
        team = frame(self.con, "SELECT *,1.0 weight FROM teams WHERE n>=2 AND n_unknown=0 AND eligible_n>0 AND ai IS NOT NULL")
        fields = ["ai", "early_share", "senior_share", "age_mean", "logq_mean", "width_mean", "first_share"]
        primary = team.loc[team.application_year <= end]
        elements = self.fit("element_scales", primary, fields, OUTCOMES[:2], "surface")[0]
        self.summarize_team(elements,"ELEMENT_SCALES")
        main, common = self.fit("main_surface", primary, fields, OUTCOMES, "surface")
        if main is None:
            raise IdentificationError("Main composition model is not estimable; see models/main_surface.json")
        self.summarize_team(main, "MAIN")
        if common is not None:
            csv(self.out / "tables/composition_support.csv", common.groupby(["ai", "n", "n_early", "n_senior", "first_share"], observed=True).size().reset_index(name="N").to_dict("records"))
            # All three outcomes use the same fixed common patent sample.
            gap = self.fit("career_distance", common, fields + ["mean_age_distance"], OUTCOMES, "distance")[0]
            if gap:
                for outcome, result in zip(OUTCOMES, gap["outcomes"]):
                    info = result["covariance"]["CPC"]
                    v = np.eye(len(gap["names"]))[gap["names"].index("career_distance_10y_AI")]
                    self.add("CONTINUOUS_DISTANCE", gap["model"], outcome, "10y_AI_difference", contrast(result["beta"], info["V"], v, info["df"], info["inference_ok"]), N=gap["N"])
            denominator = self.fit("pair_denominator", common, fields + ["pair_n"], OUTCOMES, "denominator")[0]
            self.summarize_team(denominator, "DENOMINATOR")
            linear = self.fit("linear_composition", common, fields, OUTCOMES, "linear")[0]
            self.summarize_team(linear, "FUNCTIONAL_FORM")
        self.personal(end)
        if self.cfg["robustness"]:
            self.robustness(team, fields)
        tables = {r[0] for r in self.con.execute("SHOW TABLES").fetchall()}
        if "impact" in tables:
            self.impact(team, fields)
        self.finish()

    def personal(self, end):
        columns="patent_id,inventor_id,primary_cpc,application_year,n,ai,ai_score,early_peers,senior_peers,first_peers,age,prior_patents,history_width,weight,y,stage,source_weight,utilization_weight,opportunity_n,utilization,A,B,C,D"
        data = frame(self.con, f"SELECT {columns} FROM personal WHERE weight IS NOT NULL AND application_year<={end}")
        fields = ["ai", "early_peers", "senior_peers", "first_peers", "age", "prior_patents", "history_width"]
        baseline = self.fit("personal_baseline", data, fields, ["y"], "baseline", True)[0]
        full = self.fit("personal_full", data, fields, ["y"], "full", True)[0]
        stage = self.fit("personal_stage", data, fields, ["y"], "stage", True)[0]
        if full:
            r = full["outcomes"][0]
            names = full["names"]
            for cov, info in r["covariance"].items():
                vec = np.eye(len(names))[names.index("E_AI")]
                self.add("PERSONAL", full["model"], "y", "E_AI", contrast(r["beta"], info["V"], vec, info["df"], info["inference_ok"]), covariance=cov, N=full["N"])
                restrictions = np.vstack([np.eye(len(names))[names.index(k)] for k in ["FIRSTpeer_AI", "SENIORpeer_AI"]])
                self.add("PERSONAL", full["model"], "y", "peer_AI_restrictions", joint_wald(r["beta"], info["V"], restrictions, info["df"], info["inference_ok"]), covariance=cov, N=full["N"])
        if stage:
            r = stage["outcomes"][0]
            names = stage["names"]
            for cov, info in r["covariance"].items():
                for label in ["E", "M", "S"]:
                    v = np.eye(len(names))[names.index("E_AI")]
                    if label != "M":
                        v = v + np.eye(len(names))[names.index("E_AI_stage_" + label)]
                    self.add("CAREER_STAGE", stage["model"], label, "E_AI", contrast(r["beta"], info["V"], v, info["df"], info["inference_ok"]), covariance=cov, N=stage["N"])
                restrictions = np.vstack([np.eye(len(names))[names.index("E_AI_stage_" + k)] for k in ["E", "S"]])
                self.add("CAREER_STAGE", stage["model"], "all", "stage_equality", joint_wald(r["beta"], info["V"], restrictions, info["df"], info["inference_ok"]), covariance=cov, N=stage["N"])
        src = data.loc[data.source_weight.notna()].copy()
        src["weight"] = src.source_weight
        fit = self.fit("source_composition", src, fields, ["y", "A", "B", "C", "D"], "full", True)[0]
        if fit:
            b = np.vstack([r["beta"] for r in fit["outcomes"]])
            if np.max(np.abs(b[0]-b[1:].sum(axis=0))) > 1e-8:
                raise ArithmeticError("Component coefficients do not sum to total novelty coefficients")
            for outcome, r in zip(["y", "A", "B", "C", "D"], fit["outcomes"]):
                for cov, info in r["covariance"].items():
                    v = np.eye(len(fit["names"]))[fit["names"].index("E_AI")]
                    self.add("SOURCE_COMPOSITION", fit["model"], outcome, "E_AI", contrast(r["beta"], info["V"], v, info["df"], info["inference_ok"]), covariance=cov, N=fit["N"])
        usable = data.loc[data.utilization_weight.notna()].copy()
        usable["weight"] = usable.utilization_weight
        del src
        gc.collect()
        fit = self.fit("opportunity_utilization", usable, fields + ["opportunity_n"], ["utilization"], "opportunity", True)[0]
        if fit:
            r = fit["outcomes"][0]
            for cov, info in r["covariance"].items():
                v = np.eye(len(fit["names"]))[fit["names"].index("E_AI")]
                self.add("OPPORTUNITY", fit["model"], "utilization", "E_AI", contrast(r["beta"], info["V"], v, info["df"], info["inference_ok"]), covariance=cov, N=fit["N"])
        del data, usable
        gc.collect()

    def robustness(self, team, fields):
        end = self.cfg["primary_end"]
        for early in [3, 5, 7]:
            for senior in [15, 16, 20]:
                counts = frame(self.con, f"SELECT patent_id,count(*) FILTER(WHERE age<={early})::INTEGER n_early,count(*) FILTER(WHERE age>={senior})::INTEGER n_senior,count(*) FILTER(WHERE age<={early})::DOUBLE/count(*) early_share,count(*) FILTER(WHERE age>={senior})::DOUBLE/count(*) senior_share FROM personal GROUP BY 1")
                data = team.loc[team.application_year <= end].drop(columns=["early_share", "senior_share", "n_early", "n_senior"]).merge(counts, on="patent_id", validate="one_to_one")
                name = f"threshold_E{early}_S{senior}"
                fit = self.fit(name, data, fields, OUTCOMES, "surface")[0]
                self.summarize_team(fit, "THRESHOLDS")
        for low, high in period_ranges(self.cfg):
            data = team.loc[team.application_year.between(low, high)]
            fit = self.fit(f"period_{low}_{high}", data, fields, OUTCOMES, "surface")[0]
            self.summarize_team(fit, "PERIODS")
        for w in self.cfg["windows"]:
            paired=[f"relative_full_on_w{w}",f"team_full_on_w{w}"]
            if not set(paired)<=set(team):
                raise ValueError("History-window analysis requires explicitly paired full-reference measures")
            subset_data = team.loc[team.application_year <= end].dropna(subset=paired+[f"relative_y{w}", f"team_y{w}", "pair_y"]).copy()
            for reference in ["full", str(w)]:
                data = subset_data.copy()
                if reference == "full":
                    data["relative_y"],data["team_y"]=data[paired[0]],data[paired[1]]
                else:
                    data["relative_y"], data["team_y"] = data[f"relative_y{w}"], data[f"team_y{w}"]
                fit = self.fit(f"history_{w}_{reference}", data, fields, OUTCOMES, "surface")[0]
                self.summarize_team(fit, "HISTORY_WINDOWS")
        strict = team.loc[(team.application_year <= end) & ~team.any_history_gap & (team.uncertain_new_pairs == 0)]
        fit = self.fit("strict_history", strict, fields, OUTCOMES, "surface")[0]
        self.summarize_team(fit, "HISTORY_COMPLETENESS")
        origins=frame(self.con,"""SELECT p.patent_id,min(o.first_grant) first_grant FROM personal p
          JOIN(SELECT m.inventor_id,min(t.grant_date) first_grant FROM membership m JOIN patents t USING(patent_id) GROUP BY 1)o USING(inventor_id) GROUP BY 1""")
        boundary=self.con.execute("SELECT min(grant_date)+INTERVAL '5 years' FROM patents").fetchone()[0]
        late_ids=origins.loc[origins.first_grant>=pd.Timestamp(boundary),"patent_id"]
        data=team.loc[(team.application_year<=end)&team.patent_id.isin(late_ids)]
        fit=self.fit("left_boundary_excluded",data,fields,OUTCOMES,"surface")[0]
        self.summarize_team(fit,"HISTORY_COMPLETENESS")
        for label, threshold in [("p30",.3),("p70",.7),("continuous",None)]:
            data=team.loc[team.application_year<=end].copy()
            data["ai"]=(data.ai_score>=threshold).astype(float)if threshold is not None else data.ai_score
            fit=self.fit("ai_"+label,data,fields,OUTCOMES,"surface")[0]
            self.summarize_team(fit,"AI_MEASUREMENT")
        for section in sorted(team.primary_cpc.dropna().str[0].unique()):
            data = team.loc[(team.application_year <= end) & team.primary_cpc.str.startswith(section)]
            if len(data) < self.cfg.get("min_field_rows", 500):
                self.status.append(dict(model="field_"+section,status="insufficient_support",N=len(data),reason="Below the configured field sample-size threshold"))
                continue
            fit = self.fit("field_" + section, data, fields, OUTCOMES, "surface")[0]
            self.summarize_team(fit, "FIELDS")

    def impact(self, team, fields):
        impacts = frame(self.con, "SELECT patent_id,ln(1+citations_5y) impact FROM impact")
        data = team.merge(impacts, on="patent_id", validate="one_to_one")
        try:
            data, groups = subset(data, fields + OUTCOMES + ["impact"])
            design = team_design(data, "surface")
            for name in OUTCOMES:
                design.values(name, data[name].to_numpy())
            x = np.column_stack([fn()/s for fn, s in zip(design.functions, design.scales)])
            cpc = pd.factorize(data.primary_cpc, sort=True)[0]
            fit = fit_multioutcome(x, data.impact.to_numpy(), data.weight.to_numpy(), groups,
                                   {"CPC": [cpc]}, design.names, design.scales, target="relative_y")
            fit.update(model="impact_validation", outcome_names=["impact"], synthetic=bool(self.cfg.get("synthetic")))
            self.store_coefficients(fit)
            self.status.append(dict(model="impact_validation",status="estimated",N=len(data)))
            self.samples.append(dict(model="impact_validation",N=len(data),patents=int(data.patent_id.nunique()),
                inventors=None,AI_N=int((data.ai_score>=self.cfg['ai_threshold']).sum()),FE_rank=fit["FE_rank"],weights_renormalized=False))
            save_json(self.out / "models/impact_validation.json", fit)
            r = fit["outcomes"][0]
            info = r["covariance"]["CPC"]
            for name in OUTCOMES:
                vec = np.eye(len(design.names))[design.names.index(name)]
                self.add("IMPACT_VALIDATION", "impact_validation", name, "log_5y_citations", contrast(r["beta"], info["V"], vec, info["df"], info["inference_ok"]), N=fit["N"])
        except (IdentificationError, ValueError, ArithmeticError) as error:
            self.status.append(dict(model="impact_validation", status="not_estimable", reason=str(error)))

    def finish(self):
        keys=["family","model","outcome","contrast","covariance"]
        observed={tuple(row[k]for k in keys):row for row in self.estimands}
        if len(observed)!=len(self.estimands):raise ValueError("Duplicate estimand identifiers")
        status={row["model"]:row for row in self.status}
        records=[]
        for planned in planned_tests(self.cfg):
            key=tuple(planned[k]for k in keys)
            actual=observed.pop(key,None)
            if actual is not None:
                row=dict(actual)
                row["test_status"]="estimated"if row.get("inference_ok",False)else"inference_unavailable"
            else:
                information=status.get(planned["model"],{})
                unavailable_status=information.get("status","not_run_or_insufficient_support")
                if unavailable_status=="estimated":unavailable_status="not_computed"
                row=dict(planned,p_raw=None,estimate=None,CI_low=None,CI_high=None,inference_ok=False,
                    test_status=unavailable_status,
                    reason=information.get("reason","No valid estimate was produced for this planned test"),N=information.get("N"))
            records.append(row)
        if observed:raise ValueError("Unplanned estimands: "+str(list(observed)))
        rows = pd.DataFrame(records)
        if not rows.empty:
            rows["p_Holm"] = np.nan
            for family, indices in rows.groupby("family", observed=True).groups.items():
                values = [None if pd.isna(p) else float(p) for p in rows.loc[indices, "p_raw"]]
                adjusted = holm(values)
                rows.loc[indices, "p_Holm"] = [value if raw is not None else np.nan for value,raw in zip(adjusted,values)]
                rows.loc[indices, "family_size"] = len(values)
            rows["data_kind"] = "SYNTHETIC" if self.cfg.get("synthetic") else "OBSERVED"
            rows.to_csv(self.out / "tables/estimands.csv", index=False, encoding="utf-8-sig")
        csv(self.out / "tables/sample_sizes.csv", self.samples)
        csv(self.out / "tables/coefficients.csv", self.coefficients)
        csv(self.out / "tables/model_status.csv", self.status)
        csv(self.out / "tables/profile_support.csv", self.profile_rows)
        save_json(self.out / "analysis_settings.json", self.cfg)


def analyze(con, cfg, out):
    Path(out, "tables").mkdir(exist_ok=True)
    Analysis(con, cfg, out).run()
