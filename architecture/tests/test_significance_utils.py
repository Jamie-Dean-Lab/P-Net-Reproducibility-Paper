import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

from architecture.significance_utils import (
    build_comparisons,
    corrected_paired_stats,
    load_scores_from_results,
    load_scores_from_summary,
    run_significance_tests,
)

SCORES_A = np.array([0.80, 0.84, 0.79, 0.86, 0.81])
SCORES_B = np.array([0.74, 0.79, 0.75, 0.77, 0.76])


def _naive_paired(s1, s2):
    """Uncorrected paired t-test, for comparison against the corrected one."""
    d = np.asarray(s1) - np.asarray(s2)
    n = len(d)
    sem = d.std(ddof=1) / np.sqrt(n)
    t_stat = d.mean() / sem
    return t_stat, sem, float(2 * tdist.sf(abs(t_stat), n - 1))


class TestCorrectedPairedStats(unittest.TestCase):

    def setUp(self):
        self.res = corrected_paired_stats(SCORES_A, SCORES_B)
        self.diffs = SCORES_A - SCORES_B
        self.n = len(self.diffs)
        self.sd = self.diffs.std(ddof=1)

    def test_mean_diff_is_mean_of_paired_differences(self):
        self.assertAlmostEqual(self.res["mean_diff"], self.diffs.mean(), places=12)

    def test_df_is_n_minus_one(self):
        self.assertEqual(self.res["df"], self.n - 1)

    def test_sem_matches_nadeau_bengio_formula(self):
        expected = np.sqrt((1.0 / self.n + 1.0 / (self.n - 1)) * self.sd ** 2)
        self.assertAlmostEqual(self.res["sem"], expected, places=12)

    def test_default_rho_is_one_over_k_minus_one(self):
        explicit = corrected_paired_stats(SCORES_A, SCORES_B, rho=1.0 / (self.n - 1))
        self.assertAlmostEqual(self.res["sem"], explicit["sem"], places=12)

    def test_explicit_rho_overrides_default(self):
        other = corrected_paired_stats(SCORES_A, SCORES_B, rho=0.9)
        self.assertGreater(other["sem"], self.res["sem"])

    def test_larger_rho_gives_larger_sem(self):
        sems = [corrected_paired_stats(SCORES_A, SCORES_B, rho=r)["sem"]
                for r in (0.0, 0.25, 1.0)]
        self.assertEqual(sems, sorted(sems))

    def test_rho_zero_reduces_to_naive_sem(self):
        _, naive_sem, _ = _naive_paired(SCORES_A, SCORES_B)
        self.assertAlmostEqual(
            corrected_paired_stats(SCORES_A, SCORES_B, rho=0.0)["sem"], naive_sem, places=12)

    def test_correction_is_conservative_relative_to_naive_test(self):
        """The entire point of the correction: wider SE, larger p-value."""
        naive_t, naive_sem, naive_p = _naive_paired(SCORES_A, SCORES_B)
        self.assertGreater(self.res["sem"], naive_sem)
        self.assertLess(abs(self.res["t"]), abs(naive_t))
        self.assertGreater(self.res["p_raw"], naive_p)

    def test_t_is_mean_diff_over_sem(self):
        self.assertAlmostEqual(self.res["t"], self.res["mean_diff"] / self.res["sem"], places=12)

    def test_p_raw_is_two_sided(self):
        expected = float(2 * tdist.sf(abs(self.res["t"]), self.n - 1))
        self.assertAlmostEqual(self.res["p_raw"], expected, places=12)

    def test_cohens_d_is_paired_dz(self):
        self.assertAlmostEqual(self.res["cohens_d"], self.diffs.mean() / self.sd, places=12)

    def test_ci_is_symmetric_about_mean_diff(self):
        mid = (self.res["ci_low"] + self.res["ci_high"]) / 2
        self.assertAlmostEqual(mid, self.res["mean_diff"], places=12)

    def test_ci_half_width_is_tcrit_times_sem(self):
        half = (self.res["ci_high"] - self.res["ci_low"]) / 2
        t_crit = float(tdist.ppf(0.975, self.n - 1))
        self.assertAlmostEqual(half, t_crit * self.res["sem"], places=12)

    def test_smaller_alpha_widens_ci(self):
        wide = corrected_paired_stats(SCORES_A, SCORES_B, alpha=0.01)
        self.assertLess(wide["ci_low"], self.res["ci_low"])
        self.assertGreater(wide["ci_high"], self.res["ci_high"])

    def test_alpha_does_not_affect_p_value(self):
        self.assertAlmostEqual(
            corrected_paired_stats(SCORES_A, SCORES_B, alpha=0.01)["p_raw"],
            self.res["p_raw"], places=12)

    def test_swapping_arguments_flips_sign_but_not_p(self):
        flipped = corrected_paired_stats(SCORES_B, SCORES_A)
        self.assertAlmostEqual(flipped["t"], -self.res["t"], places=12)
        self.assertAlmostEqual(flipped["mean_diff"], -self.res["mean_diff"], places=12)
        self.assertAlmostEqual(flipped["cohens_d"], -self.res["cohens_d"], places=12)
        self.assertAlmostEqual(flipped["p_raw"], self.res["p_raw"], places=12)
        self.assertAlmostEqual(flipped["sem"], self.res["sem"], places=12)

    def test_ci_excludes_zero_when_significant(self):
        self.assertLess(self.res["p_raw"], 0.05)
        self.assertGreater(self.res["ci_low"], 0.0)

    def test_ci_includes_zero_when_not_significant(self):
        noisy = np.array([0.80, 0.60, 0.90, 0.55, 0.85])
        res = corrected_paired_stats(SCORES_A, noisy)
        self.assertGreater(res["p_raw"], 0.05)
        self.assertLessEqual(res["ci_low"], 0.0)
        self.assertGreaterEqual(res["ci_high"], 0.0)

    def test_accepts_lists_as_well_as_arrays(self):
        res = corrected_paired_stats(list(SCORES_A), list(SCORES_B))
        self.assertAlmostEqual(res["t"], self.res["t"], places=12)

    def test_returns_plain_floats(self):
        for key in ("t", "mean_diff", "sem", "ci_low", "ci_high", "cohens_d", "p_raw"):
            self.assertIsInstance(self.res[key], float, msg=key)


class TestCorrectedPairedStatsDegenerate(unittest.TestCase):
    """Zero-variance differences must not abort the whole comparison run."""

    def test_identical_scores_give_no_evidence_of_difference(self):
        res = corrected_paired_stats(SCORES_A, SCORES_A)
        self.assertEqual(res["mean_diff"], 0.0)
        self.assertEqual(res["p_raw"], 1.0)
        self.assertFalse(np.isnan(res["t"]))

    def test_constant_nonzero_difference_is_maximally_significant(self):
        res = corrected_paired_stats(SCORES_A, SCORES_A - 0.05)
        self.assertAlmostEqual(res["mean_diff"], 0.05, places=12)
        self.assertEqual(res["p_raw"], 0.0)
        self.assertGreater(res["t"], 0.0)
        self.assertTrue(np.isinf(res["t"]))

    def test_constant_negative_difference_keeps_sign(self):
        res = corrected_paired_stats(SCORES_A - 0.05, SCORES_A)
        self.assertLess(res["t"], 0.0)
        self.assertEqual(res["p_raw"], 0.0)

    def test_degenerate_case_still_returns_every_key(self):
        res = corrected_paired_stats(SCORES_A, SCORES_A)
        for key in ("t", "df", "mean_diff", "sem", "ci_low", "ci_high", "cohens_d", "p_raw"):
            self.assertIn(key, res)


class TestBuildComparisons(unittest.TestCase):

    def setUp(self):
        self.disp = {"pnet": "P-NET", "a": "Model A", "b": "Model B"}.get

    def test_reference_paired_with_every_baseline(self):
        comparisons, _ = build_comparisons("pnet", ["a", "b"], self.disp)
        self.assertEqual([(c[0], c[1]) for c in comparisons], [("pnet", "a"), ("pnet", "b")])

    def test_reference_is_first_model_name(self):
        _, names = build_comparisons("pnet", ["a", "b"], self.disp)
        self.assertEqual(names, ["pnet", "a", "b"])

    def test_comparison_label_uses_display_names(self):
        comparisons, _ = build_comparisons("pnet", ["a"], self.disp)
        self.assertEqual(comparisons[0][2], "P-NET vs Model A")

    def test_empty_baselines_gives_no_comparisons(self):
        comparisons, names = build_comparisons("pnet", [], self.disp)
        self.assertEqual(comparisons, [])
        self.assertEqual(names, ["pnet"])


def _write_summary(root, model, fold, best, rows, prefix=""):
    d = os.path.join(root, model, f"test_{fold}", f"best_{best}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows, index=["train", "val", "test"]).rename(
        columns=lambda c: prefix + c).to_csv(os.path.join(d, "summary_results.csv"))


class TestLoadScoresFromSummary(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        for model, base in (("pnet", 0.80), ("dense", 0.70)):
            for fold in range(3):
                _write_summary(self.tmp, model, fold, "auc",
                               {"auc": [0.99, 0.90, base + fold / 100],
                                "f1": [0.98, 0.88, base - 0.05]},
                               prefix="response_")

    def test_reads_one_test_row_per_fold(self):
        scores = load_scores_from_summary(self.tmp, ["pnet"], "auc", "auc", "response_")
        self.assertEqual(len(scores["pnet"]), 3)

    def test_selects_test_split_not_train_or_val(self):
        scores = load_scores_from_summary(self.tmp, ["pnet"], "auc", "auc", "response_")
        self.assertTrue(np.allclose(sorted(scores["pnet"]["auc"]), [0.80, 0.81, 0.82]))

    def test_col_prefix_is_stripped(self):
        scores = load_scores_from_summary(self.tmp, ["pnet"], "auc", "auc", "response_")
        self.assertIn("auc", scores["pnet"].columns)

    def test_selects_requested_metric_column(self):
        scores = load_scores_from_summary(self.tmp, ["pnet"], "auc", "f1", "response_")
        self.assertTrue(np.allclose(scores["pnet"]["f1"], 0.75))

    def test_selection_metric_picks_the_best_directory(self):
        _write_summary(self.tmp, "pnet", 0, "f1", {"auc": [0, 0, 0.1], "f1": [0, 0, 0.2]},
                       prefix="response_")
        scores = load_scores_from_summary(self.tmp, ["pnet"], "f1", "auc", "response_")
        self.assertEqual(scores["pnet"]["auc"].tolist(), [0.1])

    def test_missing_model_directory_is_skipped(self):
        scores = load_scores_from_summary(self.tmp, ["pnet", "ghost"], "auc", "auc", "response_")
        self.assertNotIn("ghost", scores)
        self.assertIn("pnet", scores)

    def test_model_with_no_summary_files_is_omitted(self):
        os.makedirs(os.path.join(self.tmp, "empty", "test_0"), exist_ok=True)
        scores = load_scores_from_summary(self.tmp, ["empty"], "auc", "auc", "response_")
        self.assertNotIn("empty", scores)

    def test_all_models_share_one_fold_ordering(self):
        """Pairing is positional, so every model must enumerate folds identically."""
        scores = load_scores_from_summary(self.tmp, ["pnet", "dense"], "auc", "auc", "response_")
        offsets = scores["pnet"]["auc"].values - scores["dense"]["auc"].values
        self.assertTrue(np.allclose(offsets, 0.10))

    def test_double_digit_folds_stay_aligned_across_models(self):
        """sorted() orders test_10 before test_2; that must not desynchronise models."""
        root = tempfile.mkdtemp()
        for model, base in (("pnet", 0.50), ("dense", 0.40)):
            for fold in range(12):
                _write_summary(root, model, fold, "auc",
                               {"auc": [0, 0, base + fold / 100]}, prefix="response_")
        scores = load_scores_from_summary(root, ["pnet", "dense"], "auc", "auc", "response_")
        self.assertEqual(len(scores["pnet"]), 12)
        offsets = scores["pnet"]["auc"].values - scores["dense"]["auc"].values
        self.assertTrue(np.allclose(offsets, 0.10))

    def test_index_is_reset_so_pairing_is_positional(self):
        scores = load_scores_from_summary(self.tmp, ["pnet"], "auc", "auc", "response_")
        self.assertEqual(scores["pnet"].index.tolist(), [0, 1, 2])


class TestLoadScoresFromResults(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "pnet"), exist_ok=True)
        pd.DataFrame({
            "index": ["train", "val", "test"] * 3,
            "f1": [0.9, 0.8, 0.70, 0.9, 0.8, 0.71, 0.9, 0.8, 0.72],
            "auc": [0.9, 0.8, 0.60, 0.9, 0.8, 0.61, 0.9, 0.8, 0.62],
        }).to_csv(os.path.join(self.tmp, "pnet", "results.csv"))

    def test_selects_only_test_rows(self):
        scores = load_scores_from_results(self.tmp, ["pnet"], "f1")
        self.assertEqual(scores["pnet"]["f1"].tolist(), [0.70, 0.71, 0.72])

    def test_selects_requested_metric(self):
        scores = load_scores_from_results(self.tmp, ["pnet"], "auc")
        self.assertEqual(scores["pnet"]["auc"].tolist(), [0.60, 0.61, 0.62])

    def test_missing_results_file_is_skipped(self):
        scores = load_scores_from_results(self.tmp, ["pnet", "ghost"], "f1")
        self.assertEqual(list(scores), ["pnet"])

    def test_index_is_reset(self):
        scores = load_scores_from_results(self.tmp, ["pnet"], "f1")
        self.assertEqual(scores["pnet"].index.tolist(), [0, 1, 2])


class TestRunSignificanceTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.disp = {"pnet": "P-NET", "a": "Model A", "b": "Model B"}.get
        self.scores = {
            "pnet": pd.DataFrame({"auc": SCORES_A}),
            "a": pd.DataFrame({"auc": SCORES_B}),
            "b": pd.DataFrame({"auc": SCORES_A - np.array([0.002, 0.02, -0.01, 0.03, 0.0])}),
        }
        self.comparisons, _ = build_comparisons("pnet", ["a", "b"], self.disp)

    def _run(self, **kw):
        return run_significance_tests(self.scores, self.comparisons, "auc",
                                      "Test", self.tmp, self.disp, **kw)

    def test_one_row_per_comparison(self):
        self.assertEqual(len(self._run()), 2)

    def test_writes_csv_to_working_directory(self):
        self._run()
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "significance_tests.csv")))

    def test_comparisons_with_missing_models_are_dropped(self):
        comparisons, _ = build_comparisons("pnet", ["a", "ghost"], self.disp)
        res = run_significance_tests(self.scores, comparisons, "auc", "Test",
                                     self.tmp, self.disp)
        self.assertEqual(res["model2"].tolist(), ["Model A"])

    def test_fdr_correction_never_lowers_a_p_value(self):
        res = self._run()
        self.assertTrue((res["p_fdr"] >= res["p_raw"] - 1e-12).all())

    def test_significant_flag_agrees_with_fdr_and_alpha(self):
        res = self._run()
        self.assertTrue(((res["p_fdr"] <= 0.05) == res["significant"]).all())

    def test_stricter_alpha_can_only_reduce_significance(self):
        lenient = self._run(alpha=0.05)["significant"].sum()
        strict = self._run(alpha=0.001)["significant"].sum()
        self.assertLessEqual(strict, lenient)

    def test_display_names_used_in_output(self):
        res = self._run()
        self.assertEqual(res["model1"].unique().tolist(), ["P-NET"])
        self.assertIn("P-NET vs Model A", res["comparison"].tolist())

    def test_metric_column_recorded(self):
        self.assertEqual(self._run()["metric"].unique().tolist(), ["auc"])

    def test_stats_columns_present(self):
        res = self._run()
        for col in ("t", "df", "mean_diff", "sem", "ci_low", "ci_high", "cohens_d"):
            self.assertIn(col, res.columns)

    def test_rho_is_forwarded_to_the_paired_test(self):
        default_sem = self._run().loc[0, "sem"]
        wide_sem = self._run(rho=0.9).loc[0, "sem"]
        self.assertGreater(wide_sem, default_sem)

    def test_written_csv_matches_returned_frame(self):
        res = self._run()
        on_disk = pd.read_csv(os.path.join(self.tmp, "significance_tests.csv"), index_col=0)
        self.assertEqual(len(on_disk), len(res))
        self.assertEqual(on_disk["comparison"].tolist(), res["comparison"].tolist())

    def test_tied_models_do_not_abort_the_run(self):
        self.scores["b"] = pd.DataFrame({"auc": SCORES_A})
        res = self._run()
        self.assertEqual(len(res), 2)
        self.assertFalse(res["p_raw"].isna().any())


if __name__ == "__main__":
    unittest.main()
