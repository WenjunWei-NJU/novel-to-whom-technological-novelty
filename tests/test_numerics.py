"""Independent dense-dummy and hand-aggregated covariance benchmarks."""
import unittest

import numpy as np
from scipy import linalg

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "_shared"))

from study.fe import (IdentificationError, TwoWayFE, cluster_plan,
                          covariance_ok, fit_multioutcome, holm)


def dense_reference(x, y, w, a, b, clusters):
    """Full weighted least squares with redundant explicit FE dummies."""
    _, ga = np.unique(a, return_inverse=True)
    _, gb = np.unique(b, return_inverse=True)
    dummy = np.column_stack([np.eye(ga.max() + 1)[ga], np.eye(gb.max() + 1)[gb]])
    sw = np.sqrt(w)
    # Both FE blocks contain an intercept, so the full dummy matrix is rank
    # deficient. Select an independent basis before LAPACK least squares:
    # its default cutoff can otherwise retain a numerical null direction.
    weighted_dummy = sw[:, None] * dummy
    rank = np.linalg.matrix_rank(weighted_dummy)
    _, _, pivots = linalg.qr(weighted_dummy, mode="economic", pivoting=True)
    dummy = dummy[:, pivots[:rank]]
    projection = linalg.lstsq(sw[:, None] * dummy, sw[:, None] * x)[0]
    rx = x - dummy @ projection
    ry = y - dummy @ linalg.lstsq(sw[:, None] * dummy, sw * y)[0]
    beta = linalg.lstsq(sw[:, None] * rx, sw * ry)[0]
    residual = ry - rx @ beta
    bread = linalg.inv(rx.T @ (w[:, None] * rx))
    k = np.linalg.matrix_rank(np.column_stack([dummy, x]))
    meat = np.zeros((x.shape[1], x.shape[1]))
    margins = [list(g) for g in clusters]
    subsets = [(margins[0], 1)]
    if len(clusters) == 2:
        subsets += [(margins[1], 1), (list(zip(*margins)), -1)]
    for labels, sign in subsets:
        unique = sorted(set(labels))
        score = np.array([np.sum(rx[[i for i, z in enumerate(labels) if z == label]] *
                                 (w * residual)[[i for i, z in enumerate(labels) if z == label], None], axis=0)
                          for label in unique])
        factor = len(unique) / (len(unique) - 1) * (len(y) - 1) / (len(y) - k)
        meat += sign * factor * score.T @ score
    return beta, bread @ meat @ bread


class ProjectionTests(unittest.TestCase):
    def test_disconnected_weighted_FE_and_nonconsecutive_labels(self):
        rng = np.random.default_rng(901)
        a = np.tile([10, 10, 20, 20, 40, 40, 50, 50], 12)
        b = np.tile([2, 4, 2, 4, 8, 9, 8, 9], 12)
        w = rng.uniform(.05, 1, len(a))
        x = rng.normal(size=(len(a), 3))
        fe = TwoWayFE([a, b], w)
        self.assertEqual(fe.components, 2)
        self.assertEqual(fe.rank, 6)
        _, ga = np.unique(a, return_inverse=True)
        _, gb = np.unique(b, return_inverse=True)
        d = np.column_stack([np.eye(4)[ga], np.eye(4)[gb]])
        exact = x - d @ linalg.lstsq(np.sqrt(w)[:, None] * d, np.sqrt(w)[:, None] * x)[0]
        np.testing.assert_allclose(fe.project(x), exact, atol=1e-11)
        np.testing.assert_allclose(fe.project(x, "svd"), exact, atol=1e-11)
        self.assertLess(fe.orthogonality(fe.project(x)), 1e-12)

    def test_dynamic_nested_FE_degrees_of_freedom(self):
        a = np.tile(np.arange(12), 20)
        b = np.repeat(np.arange(5), 48)
        cpc = a // 3
        fe = TwoWayFE([a, b], np.ones(len(a)))
        plan = cluster_plan([cpc], fe, 3)
        self.assertEqual(fe.rank, 16)
        self.assertEqual(plan.fe_rank_counted, 4)
        self.assertEqual(plan.k, 7)
        self.assertEqual(plan.df, 3)

    def test_invalid_weights_fail(self):
        for w in [np.array([1, 0, 1]), np.array([1, np.nan, 1])]:
            with self.assertRaises(ValueError):
                TwoWayFE([np.arange(3), np.arange(3)], w)


class RegressionTests(unittest.TestCase):
    def fixture(self, seed=1729):
        rng = np.random.default_rng(seed)
        n = 600
        a = np.arange(n) % 12
        b = np.arange(n) // 40
        c1, c2 = np.arange(n) % 20, np.arange(n) // 20
        x = rng.normal(size=(n, 3))
        w = rng.uniform(.2, 1.2, n)
        common = rng.normal(size=n)
        y1 = x @ np.array([.2, -.1, .04]) + common + rng.normal(size=12)[a]
        y2 = x @ np.array([.08, .12, -.07]) + .8 * common + rng.normal(size=n) * .2
        return x, y1, y2, w, a, b, c1, c2

    def test_weighted_one_and_two_way_CRV1_against_dense_dummies(self):
        x, y, _, w, a, b, c1, c2 = self.fixture()
        for clusters in [[c1], [c1, c2]]:
            expected_b, expected_v = dense_reference(x, y, w, a, b, clusters)
            scales = np.array([1, 10, 100])
            fit = fit_multioutcome(x / scales, y, w, [a, b], {"test": clusters},
                                   ["x", "z", "IBI_AI"], scales,
                                   ssc_policy="full", target="IBI_AI")
            result = fit["outcomes"][0]
            np.testing.assert_allclose(result["beta"], expected_b, atol=1e-11)
            np.testing.assert_allclose(result["covariance"]["test"]["V"], expected_v, atol=1e-11)

    def test_between_outcome_covariance_equals_direct_difference_regression(self):
        x, y1, y2, w, a, b, c1, c2 = self.fixture(431)
        kwargs = dict(w=w, groups=[a, b], clusterings={"test": [c1, c2]},
                      names=["x", "z", "IBI_AI"], scales=[1, 1, 1], ssc_policy="full")
        fit = fit_multioutcome(x, np.column_stack([y1, y2]), **kwargs)
        diff = fit_multioutcome(x, y1 - y2, **kwargs)
        cross = fit["target_covariances"]["test"]
        variance = np.array([1, -1]) @ cross @ np.array([1, -1])
        expected = diff["outcomes"][0]["covariance"]["test"]["V"][2, 2]
        self.assertAlmostEqual(float(variance), float(expected), places=11)
        betas = [r["beta"][2] for r in fit["outcomes"]]
        self.assertAlmostEqual(betas[0] - betas[1], diff["outcomes"][0]["beta"][2], places=11)
        self.assertGreater(abs(cross[0, 1]), 1e-5)

    def test_composite_profile_covariance_uses_all_terms(self):
        x,y1,y2,w,a,b,c1,c2=self.fixture(320)
        vector=np.array([.2,-.4,.3])
        kwargs=dict(w=w,groups=[a,b],clusterings={"test":[c1,c2]},names=["x","z","IBI_AI"],scales=[1,1,1],ssc_policy="full",target_vector=vector)
        together=fit_multioutcome(x,np.column_stack([y1,y2]),**kwargs)
        difference=fit_multioutcome(x,y1-y2,**kwargs)
        v=together["target_covariances"]["test"]
        actual=np.array([1,-1])@v@np.array([1,-1])
        expected=vector@difference["outcomes"][0]["covariance"]["test"]["V"]@vector
        self.assertAlmostEqual(float(actual),float(expected),places=11)

    def test_collinear_regressor_is_blocked(self):
        x, y, _, w, a, b, c1, _ = self.fixture()
        x[:, 1] = x[:, 0]
        with self.assertRaises(IdentificationError):
            fit_multioutcome(x, y, w, [a, b], {"test": [c1]}, ["x", "z", "IBI_AI"], [1, 1, 1])

    def test_material_nonPSD_covariance_is_not_repaired(self):
        v = np.array([[1., 2.], [2., 1.]])
        original = v.copy()
        self.assertFalse(covariance_ok(v))
        np.testing.assert_array_equal(v, original)

    def test_Holm_retains_blocked_hypotheses(self):
        np.testing.assert_allclose(holm([None, .01, .04, .02]), [1, .04, .08, .06])


if __name__ == "__main__":
    unittest.main()
