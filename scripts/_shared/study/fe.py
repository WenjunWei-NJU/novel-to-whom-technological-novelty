"""Weighted two-way FE and cluster inference, with no outcome-based selection.

Two additive fixed-effect blocks are absorbed. This estimator describes
conditional associations and does not identify a causal treatment effect.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import linalg, stats
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.csgraph import connected_components


class IdentificationError(ValueError):
    """The requested regressors are not identified after absorbing the FE."""


def encode(values):
    a = np.asarray(values)
    if a.ndim != 1 or not len(a):
        raise ValueError("Group labels must be a nonempty vector")
    if a.dtype.kind in "fc" and not np.isfinite(a).all():
        raise ValueError("Missing/nonfinite group label")
    labels, codes = np.unique(a, return_inverse=True)
    return labels, codes.astype(np.int64, copy=False)


class TwoWayFE:
    """Schur-complement projection, independently checked by eigen/SVD solves.

    Only a groups-by-groups matrix is dense. The smaller FE block is used in
    the Schur solve, and input labels need not be consecutive integers.
    """

    def __init__(self, groups, weights, max_cells=25_000_000):
        if len(groups) != 2:
            raise ValueError("Exactly two additive FE blocks are required")
        self.original_groups = tuple(encode(g)[1] for g in groups)
        self.w = np.asarray(weights, dtype=float)
        if (self.w.ndim != 1 or not np.isfinite(self.w).all()
                or np.any(self.w <= 0)):
            raise ValueError("Weights must be finite and strictly positive")
        if any(len(g) != len(self.w) for g in self.original_groups):
            raise ValueError("FE labels and weights have different lengths")
        a, b = self.original_groups
        if a.max() < b.max():
            a, b = b, a
        self.a, self.b = a, b
        self.na, self.nb = int(a.max()) + 1, int(b.max()) + 1
        if self.na * self.nb > max_cells:
            raise MemoryError("FE cross-table exceeds max_fe_cells; no dense observation dummies are built")
        self.ca = np.bincount(a, weights=self.w, minlength=self.na)
        self.cb = np.bincount(b, weights=self.w, minlength=self.nb)
        self.C = np.bincount(a * self.nb + b, weights=self.w,
                             minlength=self.na * self.nb).reshape(self.na, self.nb)
        graph = csr_matrix(self.C > 0)
        self.components = int(connected_components(
            bmat([[None, graph], [graph.T, None]], format="csr"),
            directed=False, return_labels=False))
        self.rank = self.na + self.nb - self.components
        self.schur_rank = self.nb - self.components
        t = self.C / np.sqrt(self.ca[:, None] * self.cb[None, :])
        s = np.eye(self.nb) - t.T @ t
        s = (s + s.T) / 2
        ev, u = linalg.eigh(s)
        su, sv, vh = linalg.svd(s, full_matrices=False)
        r = self.schur_rank
        self.ev = ev[-r:] if r else np.empty(0)
        self.u = u[:, -r:] if r else np.empty((self.nb, 0))
        self.sv, self.su, self.v = sv[:r], su[:, :r], vh[:r].T
        if r and min(self.ev.min(), self.sv.min()) <= 100 * np.finfo(float).eps:
            raise IdentificationError("Numerically singular FE Schur system")

    def _pass(self, x, method):
        means = np.column_stack([
            np.bincount(self.a, weights=self.w * x[:, j], minlength=self.na) / self.ca
            for j in range(x.shape[1])])
        z = x - means[self.a]
        if not self.schur_rank:
            return z
        rhs = np.column_stack([
            np.bincount(self.b, weights=self.w * z[:, j], minlength=self.nb)
            for j in range(x.shape[1])]) / np.sqrt(self.cb[:, None])
        if method == "eigh":
            coef = self.u @ ((self.u.T @ rhs) / self.ev[:, None])
        else:
            coef = self.v @ ((self.su.T @ rhs) / self.sv[:, None])
        coef /= np.sqrt(self.cb[:, None])
        return z - coef[self.b] + ((self.C @ coef) / self.ca[:, None])[self.a]

    def project(self, x, method="eigh"):
        if method not in {"eigh", "svd"}:
            raise ValueError("Unknown projection method")
        x = np.asarray(x, dtype=float)
        x = x[:, None] if x.ndim == 1 else x
        if x.ndim != 2 or len(x) != len(self.w) or not np.isfinite(x).all():
            raise ValueError("Invalid projection input")
        z = self._pass(x, method)
        for _ in range(2):
            z = self._pass(z, method)
        return z

    def orthogonality(self, x):
        x = np.asarray(x)
        x = x[:, None] if x.ndim == 1 else x
        error = 0.0
        for g in self.original_groups:
            mass = np.bincount(g, weights=self.w)
            for j in range(x.shape[1]):
                error = max(error, float(np.max(np.abs(
                    np.bincount(g, weights=self.w * x[:, j]) / mass))))
        return error


def nested_in(fe, cluster):
    order = np.argsort(fe, kind="stable")
    f, c = np.asarray(fe)[order], np.asarray(cluster)[order]
    return not np.any((f[1:] == f[:-1]) & (c[1:] != c[:-1]))


def effective_fe_rank(fe, cluster_groups, policy="nested_conditional"):
    if policy == "full":
        return fe.rank
    if policy != "nested_conditional":
        raise ValueError("ssc_fe_policy must be nested_conditional or full")
    nested = [any(nested_in(g, c) for c in cluster_groups)
              for g in fe.original_groups]
    if all(nested):
        return 0
    if not any(nested):
        return fe.rank
    # Remove the nested FE span, including its overlap with the other block.
    j = nested.index(True)
    return fe.rank - len(np.unique(fe.original_groups[j]))


@dataclass
class ClusterPlan:
    parts: list
    df: int
    k: int
    n: int
    fe_rank_counted: int


def cluster_plan(groups, fe, p, policy="nested_conditional"):
    if not groups or len(groups) > 2:
        raise ValueError("Use one or two clustering dimensions")
    codes = [encode(g)[1] for g in groups]
    if any(len(g) != len(fe.w) for g in codes):
        raise ValueError("Cluster labels and observations have different lengths")
    counts = [int(g.max()) + 1 for g in codes]
    if min(counts) < 2:
        raise ValueError("Inference needs at least two clusters per margin")
    r = effective_fe_rank(fe, codes, policy)
    n, k = len(fe.w), p + r
    if n <= k:
        raise IdentificationError("No residual degrees of freedom")
    parts = []
    for size in range(1, len(codes) + 1):
        for ix in combinations(range(len(codes)), size):
            if size == 1:
                g = codes[ix[0]]
            else:
                _, g = np.unique(np.column_stack([codes[j] for j in ix]),
                                 axis=0, return_inverse=True)
            ng = int(g.max()) + 1
            factor = ng / (ng - 1) * (n - 1) / (n - k)
            parts.append(dict(codes=g, groups=ng, sign=(-1) ** (size + 1),
                              factor=factor, margins=list(ix)))
    return ClusterPlan(parts, min(counts) - 1, k, n, r)


def cluster_covariance(rx, residual, w, bread, plan, chunk=200_000):
    """Scores contain analytic weights once: sum_i w_i x_tilde_i e_i."""
    p = rx.shape[1]
    v = np.zeros((p, p))
    we = np.asarray(w) * np.asarray(residual)
    for part in plan.parts:
        g, ng = part["codes"], part["groups"]
        if ng == len(w):
            meat = np.zeros((p, p))
            for start in range(0, len(w), chunk):
                stop = min(len(w), start + chunk)
                scores = np.asarray(rx[start:stop]) * we[start:stop, None]
                meat += scores.T @ scores
        else:
            scores = np.column_stack([
                np.bincount(g, weights=np.asarray(rx[:, j]) * we, minlength=ng)
                for j in range(p)])
            meat = scores.T @ scores
        v += part["sign"] * part["factor"] * (bread @ meat @ bread)
    return (v + v.T) / 2


def covariance_ok(v):
    if not np.isfinite(v).all() or np.any(np.diag(v) <= 0):
        return False
    ev = linalg.eigvalsh((v + v.T) / 2)
    corr = v / np.sqrt(np.outer(np.diag(v), np.diag(v)))
    ce = linalg.eigvalsh((corr + corr.T) / 2)
    tol = 100 * np.finfo(float).eps * len(v)
    return bool(ev[0] >= -tol * max(abs(ev))
                and ce[0] >= -tol * max(abs(ce)))


def contrast(beta, covariance, vector, df, inference_ok=True):
    vector = np.asarray(vector, dtype=float)
    value = float(vector @ beta)
    variance = float(vector @ covariance @ vector)
    result = dict(estimate=value, variance=variance, SE=None, CI_low=None,
                  CI_high=None, p_raw=None, df=int(df), inference_ok=False)
    if inference_ok and np.isfinite(variance) and variance > 0:
        se = float(np.sqrt(variance))
        critical = float(stats.t.ppf(.975, df))
        result.update(SE=se, CI_low=value - critical * se,
                      CI_high=value + critical * se,
                      p_raw=float(2 * stats.t.sf(abs(value / se), df)),
                      inference_ok=True)
    return result


def joint_wald(beta, covariance, restrictions, df, inference_ok=True):
    r = np.asarray(restrictions, dtype=float)
    result = dict(F=None, p_raw=None, df_numerator=len(r),
                  df_denominator=int(df), inference_ok=False)
    if not inference_ok:
        return result
    v = r @ covariance @ r.T
    v = (v + v.T) / 2
    if len(r) != np.linalg.matrix_rank(v) or linalg.eigvalsh(v)[0] <= 0:
        return result
    z = r @ beta
    f = float(z @ linalg.solve(v, z, assume_a="pos") / len(r))
    result.update(F=f, p_raw=float(stats.f.sf(f, len(r), df)), inference_ok=True)
    return result


def holm(pvalues):
    """Missing/blocked hypotheses stay in the planned family, conservatively as 1."""
    ps = np.asarray([1.0 if p is None else p for p in pvalues], dtype=float)
    if not np.isfinite(ps).all() or np.any((ps < 0) | (ps > 1)):
        raise ValueError("Invalid p value")
    order = np.argsort(ps, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate(
        (len(ps) - np.arange(len(ps))) * ps[order]))
    output = np.empty(len(ps))
    output[order] = adjusted
    return output.tolist()


def fit_multioutcome(x, y, w, groups, clusterings, names, scales,
                     cache=None, ssc_policy="nested_conditional", chunk=200_000,
                     max_fe_cells=25_000_000, target="IBI_AI", target_vector=None):
    """Fit multiple outcomes on exactly the same X, rows, weights, and FE.

    RX is a memory map when cache is supplied. Cluster cross-covariances of
    the target coefficient are retained for a direct between-outcome contrast.
    """
    x, y, w = np.asarray(x), np.asarray(y), np.asarray(w, dtype=float)
    y = y[:, None] if y.ndim == 1 else y
    n, p = x.shape
    scales = np.asarray(scales, dtype=float)
    if y.shape[0] != n or len(w) != n or len(names) != p or scales.shape != (p,):
        raise ValueError("Inconsistent model dimensions")
    if (not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(scales).all() or np.any(scales <= 0)):
        raise ValueError("Nonfinite model data/scales")
    fe = TwoWayFE(groups, w, max_fe_cells)
    if cache is not None:
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        rx = np.lib.format.open_memmap(cache / "RX.npy", mode="w+", dtype="float64", shape=x.shape)
    else:
        rx = np.empty_like(x, dtype=float)
    checks = []
    for j in range(0, p, 2):
        stop = min(p, j + 2)
        r = fe.project(x[:, j:stop])
        ri = fe.project(x[:, j:stop], "svd")
        error = fe.orthogonality(r * scales[j:stop])
        difference = float(np.max(np.abs((r - ri) * scales[j:stop])))
        if error > 1e-8 or difference > 1e-7:
            raise ArithmeticError("FE projection failed independent verification")
        checks.append(dict(columns=names[j:stop], orthogonality=error,
                           independent_difference=difference))
        rx[:, j:stop] = r
    if hasattr(rx, "flush"):
        rx.flush()
    gram = np.zeros((p, p))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        xx = np.asarray(rx[start:stop])
        gram += xx.T @ (w[start:stop, None] * xx)
    sd = np.sqrt(np.diag(gram))
    if np.any(sd <= 1e-12):
        raise IdentificationError("One or more regressors are absorbed by the fixed effects")
    corr = gram / np.outer(sd, sd)
    if np.linalg.matrix_rank(corr, tol=1e-9) != p:
        raise IdentificationError("Requested regressors are collinear; no automatic column removal")
    eig = linalg.eigvalsh(corr)
    if eig[0] <= 0:
        raise IdentificationError("Regressor Gram matrix is not positive definite")
    chol = linalg.cho_factor(corr, lower=True)
    bread = linalg.cho_solve(chol, np.eye(p)) / np.outer(sd, sd)
    plans = {key: cluster_plan(c, fe, p, ssc_policy) for key, c in clusterings.items()}
    targets = np.zeros((n, y.shape[1]))
    outcomes = []
    vector = np.eye(p)[names.index(target)] if target_vector is None else np.asarray(target_vector, dtype=float)
    if vector.shape != (p,):
        raise ValueError("Target contrast vector has the wrong dimension")
    direction = bread @ (vector / scales)
    for j in range(y.shape[1]):
        yy = y[:, j]
        ry = fe.project(yy)[:, 0]
        ryi = fe.project(yy, "svd")[:, 0]
        if fe.orthogonality(ry) > 1e-8 or np.max(np.abs(ry - ryi)) > 1e-7:
            raise ArithmeticError("Outcome projection failed independent verification")
        rhs = np.zeros(p)
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            rhs += np.asarray(rx[start:stop]).T @ (w[start:stop] * ry[start:stop])
        bs = linalg.cho_solve(chol, rhs / sd) / sd
        # Independent normalized eigensolve checks coefficient scaling as well.
        ev, u = linalg.eigh(corr)
        independent = (u @ ((u.T @ (rhs / sd)) / ev)) / sd
        if not np.allclose(bs / scales, independent / scales, atol=1e-8, rtol=1e-7):
            raise ArithmeticError("Coefficient solvers disagree")
        residual = np.empty(n)
        gradient = np.zeros(p)
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            xx = np.asarray(rx[start:stop])
            residual[start:stop] = ry[start:stop] - xx @ bs
            we = w[start:stop] * residual[start:stop]
            gradient += xx.T @ we
            targets[start:stop, j] = (xx @ direction) * we
        normal_error = float(np.max(np.abs(gradient) / (sd * max(1.0, np.sqrt(np.dot(w, residual ** 2))))))
        if normal_error > 1e-8 or fe.orthogonality(residual) > 1e-8:
            raise ArithmeticError("Normal equations or residual FE orthogonality failed")
        covariances = {}
        for key, plan in plans.items():
            vs = cluster_covariance(rx, residual, w, bread, plan, chunk)
            v = vs / np.outer(scales, scales)
            covariances[key] = dict(V=v, inference_ok=covariance_ok(v), df=plan.df,
                                    K=plan.k, fe_rank_counted=plan.fe_rank_counted,
                                    components=[{k: z for k, z in part.items() if k != "codes"}
                                                for part in plan.parts])
        sst = float(np.dot(w, ry ** 2))
        mean = float(np.average(yy, weights=w))
        pooled_sd = float(np.sqrt(np.average((yy - mean) ** 2, weights=w)))
        outcomes.append(dict(beta=bs / scales, covariance=covariances,
                             weighted_mean=mean, weighted_population_SD=pooled_sd,
                             zero_share=float(np.mean(yy == 0)),
                             within_R2=1 - float(np.dot(w, residual ** 2)) / sst if sst > 0 else None,
                             normal_equation_error=normal_error))
    target_covariances = {}
    for key, plan in plans.items():
        cross = np.zeros((y.shape[1], y.shape[1]))
        for part in plan.parts:
            g, ng = part["codes"], part["groups"]
            score = np.column_stack([np.bincount(g, weights=targets[:, j], minlength=ng)
                                     for j in range(y.shape[1])])
            cross += part["sign"] * part["factor"] * (score.T @ score)
        for j, out in enumerate(outcomes):
            reference = vector @ out["covariance"][key]["V"] @ vector
            if not np.isclose(cross[j, j], reference, atol=1e-12, rtol=1e-7):
                raise ArithmeticError("Target cross-outcome covariance diagonal disagrees with full covariance")
        target_covariances[key] = (cross + cross.T) / 2
    return dict(names=names, N=n, p=p, FE_rank=fe.rank, FE_components=fe.components,
                normalized_gram_condition=float(eig[-1] / eig[0]), projection_checks=checks,
                outcomes=outcomes, target_covariances=target_covariances,
                target_vector=vector,
                cluster_df={k: z.df for k, z in plans.items()}, ssc_policy=ssc_policy,
                CI_type="pointwise 95% CRV1 t interval; not Holm adjusted")
