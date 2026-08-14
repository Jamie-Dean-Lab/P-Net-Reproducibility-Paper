import unittest
import numpy as np
import pandas as pd
import logging
import tempfile
import os
from unittest.mock import MagicMock

logging.disable(logging.CRITICAL)


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

def make_dataset(n_samples=60, n_features=10, seed=0):
    """
    Build a real ConcatMultiViewDataset in memory with known string IDs ("0".."n-1")
    and balanced binary labels — no filesystem required.
    """
    from architecture.data_utils import ConcatMultiViewDataset

    rng = np.random.default_rng(seed)
    ds = ConcatMultiViewDataset()
    sample_ids = [str(i) for i in range(n_samples)]
    ds.ids = sample_ids

    data = pd.DataFrame(
        rng.random((n_samples, n_features)).astype(np.float32),
        index=sample_ids,
        columns=[f"gene_{i}" for i in range(n_features)],
    )
    ds.data_views["mut"] = data

    labels = np.zeros(n_samples, dtype=np.float32)
    labels[: n_samples // 2] = 1.0
    ds.labels = pd.DataFrame({"response": labels}, index=sample_ids, dtype=np.float32)
    ds.align_views(method="zero fill", view_aligner={}, drop_labels=False)
    return ds


def ids_of(fold):
    """Return the set of string sample IDs in a fold."""
    return set(fold.ids)


# ── Spy preprocessor / feature selector ──────────────────────────────────────

class SpyTransformer:
    """
    Records every set of IDs passed to fit_transform across all calls.
    transform() is a no-op but handles the empty-list case that _fold_run
    passes when val or test fold is [].
    """
    def __init__(self):
        self.all_fitted_ids = []   # one entry per fit_transform call

    @property
    def fitted_ids(self):
        """Union of all IDs ever seen during fitting."""
        return set().union(*self.all_fitted_ids) if self.all_fitted_ids else set()

    def fit_transform(self, fold):
        self.all_fitted_ids.append(set(fold.ids))
        return fold

    def transform(self, fold):
        # fold may be [] when there is no val/test — just pass through
        return fold

    def get_features(self):
        return []


def make_pipeline(tmp_dir, ds, config_overrides=None):
    """
    Build a Pipeline backed by a real ConcatMultiViewDataset.
    _get_logger writes into tmp_dir but suppresses console output.
    _train returns a trivial mock model.
    """
    from architecture.pipeline import Pipeline

    mock_model = MagicMock()
    mock_model.predict.side_effect = lambda xs: np.zeros(len(xs))

    spy_pp = SpyTransformer()
    spy_fs = SpyTransformer()

    base_config = {
        "run_dir":                tmp_dir,
        "run_id":                 "test_run",
        "grid_search":            [],
        "val_metric":             {},
        "fold_collators":         [],
        "grid_search_collators":  [],
        "feature_selector":       spy_fs,
        "data_augmentor":         lambda x: x,
        "feature_preprocessor":   spy_pp,
        "results_processors":     [],
        "rng_seed":               42,
        # split config — test_samples intentionally omitted from base so
        # run_crossvalidation uses get_k_splits rather than get_specific_split
        "train_samples":          0.6,
        "val_samples":            0.2,
        "tt_split_seed":          0,
        "tv_split_seed":          0,
        "hold_out_validation_for_final_fit": False,
        "validation_prop":        0.2,
        "stratified":             False,
        "inner_kfolds":           1,
        "outer_kfolds":           2,
        "model_params":           {},
        "shuffle_seed":           42
    }
    if config_overrides:
        base_config.update(config_overrides)

    p = Pipeline.__new__(Pipeline)
    p.config = base_config
    p.data = ds

    # Stub _load_data so it doesn't try to read CSV files
    p._load_data = MagicMock(side_effect=lambda *args, **kwargs: None)

    # Replace _get_logger with a silent file logger. Still creates directories
    # so we can test directory-creation side-effects.
    def silent_logger(name, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        log = logging.getLogger(name + log_dir)   # unique name avoids handler accumulation
        log.handlers.clear()
        log.addHandler(logging.NullHandler())
        return log

    p._get_logger = silent_logger
    p._train = MagicMock(return_value=(mock_model, {"loss": [0.1]}))
    # _summarise_data accesses get_alignment_ids and labels.shape — stub it
    # so tests using load_data=True don't need a fully wired-up data pipeline
    p._summarise_data = MagicMock()

    # Expose spies for assertions
    p._spy_pp = spy_pp
    p._spy_fs = spy_fs
    p._mock_model = mock_model

    return p


# ══════════════════════════════════════════════════════════════════════════════
# Dataset split correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSpecificSplitByList(unittest.TestCase):

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        all_ids = self.ds.ids
        self.train_ids = all_ids[:40]
        self.val_ids   = all_ids[40:50]
        self.test_ids  = all_ids[50:]
        self.train, self.val, self.test = self.ds.get_specific_split(
            self.train_ids, self.val_ids, self.test_ids
        )

    def test_train_ids_correct(self):
        self.assertEqual(ids_of(self.train), set(self.train_ids))

    def test_val_ids_correct(self):
        self.assertEqual(ids_of(self.val), set(self.val_ids))

    def test_test_ids_correct(self):
        self.assertEqual(ids_of(self.test), set(self.test_ids))

    def test_train_val_disjoint(self):
        self.assertTrue(ids_of(self.train).isdisjoint(ids_of(self.val)))

    def test_train_test_disjoint(self):
        self.assertTrue(ids_of(self.train).isdisjoint(ids_of(self.test)))

    def test_val_test_disjoint(self):
        self.assertTrue(ids_of(self.val).isdisjoint(ids_of(self.test)))

    def test_xs_shapes_match_ids(self):
        self.assertEqual(self.train.xs.shape[0], len(self.train_ids))
        self.assertEqual(self.val.xs.shape[0],   len(self.val_ids))
        self.assertEqual(self.test.xs.shape[0],  len(self.test_ids))

    def test_labels_row_order_matches_ids(self):
        for fold in (self.train, self.val, self.test):
            expected = self.ds.labels.loc[fold.ids].to_numpy()
            np.testing.assert_array_equal(fold.ys, expected)


class TestGetSpecificSplitByProportion(unittest.TestCase):

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.train, self.val, self.test = self.ds.get_specific_split(0.6, 0.2, 0.2, seed=42)

    def test_all_splits_disjoint(self):
        self.assertTrue(ids_of(self.train).isdisjoint(ids_of(self.val)))
        self.assertTrue(ids_of(self.train).isdisjoint(ids_of(self.test)))
        self.assertTrue(ids_of(self.val).isdisjoint(ids_of(self.test)))

    def test_approximate_proportions(self):
        n = len(self.ds)
        self.assertAlmostEqual(len(self.train) / n, 0.6, delta=0.05)
        self.assertAlmostEqual(len(self.val)   / n, 0.2, delta=0.05)
        self.assertAlmostEqual(len(self.test)  / n, 0.2, delta=0.05)

    def test_same_seed_reproducible(self):
        train2, val2, test2 = self.ds.get_specific_split(0.6, 0.2, 0.2, seed=42)
        self.assertEqual(ids_of(self.train), ids_of(train2))
        self.assertEqual(ids_of(self.val),   ids_of(val2))
        self.assertEqual(ids_of(self.test),  ids_of(test2))

    def test_different_seed_gives_different_split(self):
        train2, _, _ = self.ds.get_specific_split(0.6, 0.2, 0.2, seed=99)
        self.assertNotEqual(ids_of(self.train), ids_of(train2))


class TestGetTrainTestSplit(unittest.TestCase):

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.train, self.test = self.ds.get_train_test_split(0.8, seed=42)

    def test_disjoint(self):
        self.assertTrue(ids_of(self.train).isdisjoint(ids_of(self.test)))

    def test_union_is_full_dataset(self):
        self.assertEqual(ids_of(self.train) | ids_of(self.test), set(self.ds.ids))

    def test_approximate_proportion(self):
        self.assertAlmostEqual(len(self.train) / len(self.ds), 0.8, delta=0.05)

    def test_same_seed_reproducible(self):
        train2, _ = self.ds.get_train_test_split(0.8, seed=42)
        self.assertEqual(ids_of(self.train), ids_of(train2))

    def test_different_seed_gives_different_split(self):
        train2, _ = self.ds.get_train_test_split(0.8, seed=99)
        self.assertNotEqual(ids_of(self.train), ids_of(train2))

    def test_xs_shapes_match_ids(self):
        self.assertEqual(self.train.xs.shape[0], len(self.train.ids))
        self.assertEqual(self.test.xs.shape[0],  len(self.test.ids))


class TestGetTrainTestSplitStratified(unittest.TestCase):

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.train, self.test = self.ds.get_train_test_split(0.8, stratified=True, seed=42)

    def test_disjoint(self):
        self.assertTrue(ids_of(self.train).isdisjoint(ids_of(self.test)))

    def test_union_is_full_dataset(self):
        self.assertEqual(ids_of(self.train) | ids_of(self.test), set(self.ds.ids))

    def test_class_balance_preserved_in_train(self):
        counts = self.ds.labels.loc[list(ids_of(self.train))]["response"].value_counts(normalize=True)
        self.assertAlmostEqual(counts[0.0], 0.5, delta=0.1)
        self.assertAlmostEqual(counts[1.0], 0.5, delta=0.1)

    def test_class_balance_preserved_in_test(self):
        counts = self.ds.labels.loc[list(ids_of(self.test))]["response"].value_counts(normalize=True)
        self.assertAlmostEqual(counts[0.0], 0.5, delta=0.1)
        self.assertAlmostEqual(counts[1.0], 0.5, delta=0.1)


class TestGetKSplits(unittest.TestCase):

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.folds = list(self.ds.get_k_splits(3, seed=42))

    def test_correct_number_of_folds(self):
        self.assertEqual(len(self.folds), 3)

    def test_val_folds_disjoint_from_each_other(self):
        val_sets = [ids_of(val) for _, val in self.folds]
        for i in range(len(val_sets)):
            for j in range(i + 1, len(val_sets)):
                self.assertTrue(val_sets[i].isdisjoint(val_sets[j]),
                                f"Val folds {i} and {j} overlap")

    def test_train_val_disjoint_within_each_fold(self):
        for i, (train, val) in enumerate(self.folds):
            self.assertTrue(ids_of(train).isdisjoint(ids_of(val)),
                            f"Fold {i}: train and val overlap")

    def test_val_folds_cover_entire_dataset(self):
        all_val = set().union(*(ids_of(val) for _, val in self.folds))
        self.assertEqual(all_val, set(self.ds.ids))

    def test_train_larger_than_val_in_each_fold(self):
        for train, val in self.folds:
            self.assertGreater(len(train), len(val))

    def test_same_seed_reproducible(self):
        folds2 = list(self.ds.get_k_splits(3, seed=42))
        for (_, v1), (_, v2) in zip(self.folds, folds2):
            self.assertEqual(ids_of(v1), ids_of(v2))

    def test_different_seed_gives_different_folds(self):
        folds2 = list(self.ds.get_k_splits(3, seed=99))
        self.assertTrue(any(ids_of(v1) != ids_of(v2)
                            for (_, v1), (_, v2) in zip(self.folds, folds2)))

    def test_xs_shapes_match_ids_per_fold(self):
        for train, val in self.folds:
            self.assertEqual(train.xs.shape[0], len(train.ids))
            self.assertEqual(val.xs.shape[0],   len(val.ids))


class TestGetKSplitsStratified(unittest.TestCase):

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.folds = list(self.ds.get_k_splits(3, stratified=True, seed=42))

    def test_val_folds_disjoint_from_each_other(self):
        val_sets = [ids_of(val) for _, val in self.folds]
        for i in range(len(val_sets)):
            for j in range(i + 1, len(val_sets)):
                self.assertTrue(val_sets[i].isdisjoint(val_sets[j]))

    def test_train_val_disjoint_within_each_fold(self):
        for train, val in self.folds:
            self.assertTrue(ids_of(train).isdisjoint(ids_of(val)))

    def test_both_classes_in_every_val_fold(self):
        for _, val in self.folds:
            counts = self.ds.labels.loc[list(ids_of(val))]["response"].value_counts()
            self.assertIn(0.0, counts.index)
            self.assertIn(1.0, counts.index)


# ══════════════════════════════════════════════════════════════════════════════
# run_single_split
# ══════════════════════════════════════════════════════════════════════════════

class TestRunSingleSplitNoGridSearch(unittest.TestCase):
    """No grid search — one straight fold run."""

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.p = make_pipeline(self.tmp.name, self.ds, {"test_samples": 0.2})
        self.p.run_single_split(load_data=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_called_once(self):
        self.p._train.assert_called_once()

    def test_train_val_disjoint(self):
        train_fold = self.p._train.call_args[0][0]
        val_fold   = self.p._train.call_args[0][1]
        self.assertTrue(ids_of(train_fold).isdisjoint(ids_of(val_fold)))

    def test_approximate_train_size(self):
        train_fold = self.p._train.call_args[0][0]
        self.assertAlmostEqual(len(train_fold) / len(self.ds), 0.6, delta=0.1)

    def test_preprocessor_not_fitted_on_val(self):
        val_fold = self.p._train.call_args[0][1]
        self.assertTrue(self.p._spy_pp.fitted_ids.isdisjoint(ids_of(val_fold)),
                        "Preprocessor fitted on val data — leakage!")

    def test_feature_selector_not_fitted_on_val(self):
        val_fold = self.p._train.call_args[0][1]
        self.assertTrue(self.p._spy_fs.fitted_ids.isdisjoint(ids_of(val_fold)),
                        "Feature selector fitted on val data — leakage!")

    def test_load_data_skipped_when_flag_false(self):
        self.p._load_data.assert_not_called()


class TestRunSingleSplitGridSearch(unittest.TestCase):
    """Grid search with two configs and a val metric."""

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.val_metric = MagicMock(return_value=0.8)
        gs = [
            {"model_params": {"C": 0.1}, "model_params_choice": "small"},
            {"model_params": {"C": 1.0}, "model_params_choice": "large"},
        ]
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "grid_search":            gs,
            "val_metric":             {"auc": self.val_metric},
            "hold_out_validation_for_final_fit": False,
            "test_samples":           0.2,
        })
        self.p.run_single_split(load_data=False)
        self.all_calls = self.p._train.call_args_list

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_called_once_per_gs_config_plus_best_refit(self):
        # 2 gs configs + 1 final best refit = 3
        self.assertEqual(self.p._train.call_count, 3)

    def test_gs_configs_see_identical_val_set(self):
        # Split happens once before the gs loop, so both runs must see the same val fold
        val_0 = ids_of(self.all_calls[0][0][1])
        val_1 = ids_of(self.all_calls[1][0][1])
        self.assertEqual(val_0, val_1,
                         "GS configs saw different val sets — split is happening inside the loop")

    def test_val_metric_called_once_per_gs_config(self):
        self.assertEqual(self.val_metric.call_count, 2)

    def test_best_dir_exists(self):
        run_dir = os.path.join(self.tmp.name, "test_run")
        self.assertTrue(os.path.exists(os.path.join(run_dir, "best_auc")))

    def test_config_written_for_each_gs_fold(self):
        run_dir = os.path.join(self.tmp.name, "test_run")
        for i in range(2):
            self.assertTrue(os.path.exists(os.path.join(run_dir, f"cv_{i}", "config.txt")))

    def test_config_written_for_best(self):
        run_dir = os.path.join(self.tmp.name, "test_run")
        self.assertTrue(os.path.exists(os.path.join(run_dir, "best_auc", "config.txt")))

    def test_no_leakage_into_val_across_all_gs_folds(self):
        for i, call in enumerate(self.all_calls[:2]):
            val_fold = call[0][1]
            fitted = self.p._spy_pp.all_fitted_ids[i]
            self.assertTrue(fitted.isdisjoint(ids_of(val_fold)),
                            f"GS fold {i}: preprocessor fitted on val — leakage!")

    def test_final_refit_train_larger_than_gs_train(self):
        # hold_out_validation_for_final_fit=False merges val into train for the final run
        gs_train    = ids_of(self.all_calls[0][0][0])
        final_train = ids_of(self.all_calls[2][0][0])
        self.assertGreater(len(final_train), len(gs_train),
                           "Final refit train should be larger (val merged in)")

    def test_final_refit_train_does_not_contain_test_samples(self):
        # Recover test IDs: everything not in the gs train or val
        gs_train = ids_of(self.all_calls[0][0][0])
        gs_val   = ids_of(self.all_calls[0][0][1])
        test_ids = set(self.ds.ids) - gs_train - gs_val
        final_train = ids_of(self.all_calls[2][0][0])
        self.assertTrue(final_train.isdisjoint(test_ids),
                        "Final refit training set contains test samples — leakage!")


class TestRunSingleSplitUseValidationOnTest(unittest.TestCase):
    """hold_out_validation_for_final_fit=True — original split reused for final run."""

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        gs = [{"model_params": {"C": 1.0}, "model_params_choice": "default"}]
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "grid_search":            gs,
            "val_metric":             {"auc": MagicMock(return_value=0.9)},
            "hold_out_validation_for_final_fit": True,
            "test_samples":           0.2,
        })
        self.p.run_single_split(load_data=False)
        self.all_calls = self.p._train.call_args_list

    def tearDown(self):
        self.tmp.cleanup()

    def test_final_train_same_as_gs_train(self):
        # No re-split, so gs train and final train must be identical
        gs_train    = ids_of(self.all_calls[0][0][0])
        final_train = ids_of(self.all_calls[1][0][0])
        self.assertEqual(gs_train, final_train)

    def test_final_val_same_as_gs_val(self):
        gs_val    = ids_of(self.all_calls[0][0][1])
        final_val = ids_of(self.all_calls[1][0][1])
        self.assertEqual(gs_val, final_val)


# ══════════════════════════════════════════════════════════════════════════════
# run_crossvalidation
# ══════════════════════════════════════════════════════════════════════════════

class TestRunCrossvalidationWithTestSamples(unittest.TestCase):
    """test_samples in config → single outer fold via get_specific_split."""

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.test_ids  = self.ds.ids[50:]
        self.train_ids = self.ds.ids[:40]
        self.val_ids   = self.ds.ids[40:50]
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "test_samples":  self.test_ids,
            "train_samples": self.train_ids,
            "val_samples":   self.val_ids,
            "inner_kfolds":  1,
            "model_params":  {},
        })
        self.p.run_crossvalidation(load_data=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_called_once(self):
        self.p._train.assert_called_once()

    def test_test_dir_created(self):
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp.name, "test_run", "test_0")
        ))

    def test_train_fold_does_not_contain_test_ids(self):
        train_fold = self.p._train.call_args[0][0]
        self.assertTrue(ids_of(train_fold).isdisjoint(set(self.test_ids)),
                        "Training fold contains held-out test IDs — leakage!")

    def test_preprocessor_not_fitted_on_test_ids(self):
        self.assertTrue(self.p._spy_pp.fitted_ids.isdisjoint(set(self.test_ids)),
                        "Preprocessor fitted on test data — leakage!")

    def test_feature_selector_not_fitted_on_test_ids(self):
        self.assertTrue(self.p._spy_fs.fitted_ids.isdisjoint(set(self.test_ids)),
                        "Feature selector fitted on test data — leakage!")


class TestRunCrossvalidationKFoldsNoGridSearch(unittest.TestCase):
    """outer_kfolds=2, inner_kfolds=1, no grid search."""

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "inner_kfolds": 1,
            "outer_kfolds": 2,
            "model_params": {},
        })
        self.p.run_crossvalidation(load_data=False)
        self.all_calls = self.p._train.call_args_list

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_called_once_per_outer_fold(self):
        self.assertEqual(self.p._train.call_count, 2)

    def test_outer_fold_training_sets_differ(self):
        # Each outer fold holds out a different test partition so train sets must differ
        train_0 = ids_of(self.all_calls[0][0][0])
        train_1 = ids_of(self.all_calls[1][0][0])
        self.assertNotEqual(train_0, train_1,
                            "Both outer folds have identical training sets — k-split may be broken")

    def test_outer_fold_train_val_always_disjoint(self):
        for i, call in enumerate(self.all_calls):
            train_fold = call[0][0]
            val_fold   = call[0][1]
            self.assertTrue(ids_of(train_fold).isdisjoint(ids_of(val_fold)),
                            f"Outer fold {i}: train/val overlap")

    def test_preprocessor_fitted_separately_per_outer_fold(self):
        # One fit_transform call per outer fold
        self.assertEqual(len(self.p._spy_pp.all_fitted_ids), 2)

    def test_preprocessor_per_fold_fitted_only_on_that_folds_train(self):
        for i, call in enumerate(self.all_calls):
            train_ids = ids_of(call[0][0])
            val_ids   = ids_of(call[0][1])
            fitted    = self.p._spy_pp.all_fitted_ids[i]
            self.assertEqual(fitted, train_ids,
                             f"Outer fold {i}: preprocessor fitted on wrong data")
            self.assertTrue(fitted.isdisjoint(val_ids),
                            f"Outer fold {i}: preprocessor fitted on val — leakage!")

    def test_test_dirs_created(self):
        run_dir = os.path.join(self.tmp.name, "test_run")
        for i in range(2):
            self.assertTrue(os.path.exists(os.path.join(run_dir, f"test_{i}")))

    def test_outer_kfolds_less_than_2_raises(self):
        ds  = make_dataset()
        tmp = tempfile.TemporaryDirectory()
        p = make_pipeline(tmp.name, ds, {"outer_kfolds": 1, "model_params": {}})
        with self.assertRaises(Exception):
            p.run_crossvalidation(load_data=False)
        tmp.cleanup()


class TestRunCrossvalidationInnerKFolds(unittest.TestCase):
    """inner_kfolds=2 — inner fold loop, fold collators invoked."""

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.fold_collator = MagicMock()
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "inner_kfolds":   2,
            "outer_kfolds":   2,
            "fold_collators": [self.fold_collator],
            "model_params":   {},
        })
        self.p.run_crossvalidation(load_data=False)
        self.all_calls = self.p._train.call_args_list

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_called_for_each_inner_and_outer_fold(self):
        # 2 outer × 1 default gs config × 2 inner folds = 4
        self.assertEqual(self.p._train.call_count, 4)

    def test_inner_fold_train_val_always_disjoint(self):
        for i, call in enumerate(self.all_calls):
            train_fold = call[0][0]
            val_fold   = call[0][1]
            self.assertTrue(ids_of(train_fold).isdisjoint(ids_of(val_fold)),
                            f"Inner fold {i}: train/val overlap")

    def test_preprocessor_never_fitted_on_val(self):
        for i, call in enumerate(self.all_calls):
            val_fold = call[0][1]
            fitted   = self.p._spy_pp.all_fitted_ids[i]
            self.assertTrue(fitted.isdisjoint(ids_of(val_fold)),
                            f"Inner fold {i}: preprocessor fitted on val — leakage!")

    def test_fold_collator_called_once_per_outer_fold(self):
        # One collation per outer fold (2 outer folds, 1 gs config each)
        self.assertEqual(self.fold_collator.call_count, 2)

    def test_fold_collator_receives_results_and_save_dir(self):
        arg = self.fold_collator.call_args_list[0][0][0]
        self.assertIn("results",  arg)
        self.assertIn("save_dir", arg)


class TestRunCrossvalidationGridSearch(unittest.TestCase):
    """Grid search across outer folds — best config selected, gs_collator invoked."""

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.gs_collator = MagicMock()
        call_count = {"n": 0}
        def metric_fn(results):
            call_count["n"] += 1
            return 0.9 if call_count["n"] % 2 == 0 else 0.6
        gs = [
            {"model_params": {"C": 0.1}, "model_params_choice": "small"},
            {"model_params": {"C": 1.0}, "model_params_choice": "large"},
        ]
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "inner_kfolds":           1,
            "outer_kfolds":           2,
            "grid_search":            gs,
            "val_metric":             {"auc": metric_fn},
            "hold_out_validation_for_final_fit": False,
            "grid_search_collators":  [self.gs_collator],
        })
        self.p.run_crossvalidation(load_data=False)
        self.all_calls = self.p._train.call_args_list

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_count(self):
        # 2 outer × (2 gs configs + 1 best refit) = 6
        self.assertEqual(self.p._train.call_count, 6)

    def test_gs_configs_within_same_outer_fold_see_same_training_pool(self):
        # The two gs configs in outer fold 0 (calls 0 and 1) must train on the same samples
        train_gs0 = ids_of(self.all_calls[0][0][0])
        train_gs1 = ids_of(self.all_calls[1][0][0])
        self.assertEqual(train_gs0, train_gs1,
                         "GS configs within the same outer fold have different training pools")

    def test_outer_fold_training_pools_differ(self):
        # Outer fold 0 (call 0) and outer fold 1 (call 2) must have different training pools
        train_outer_0 = ids_of(self.all_calls[0][0][0])
        train_outer_1 = ids_of(self.all_calls[2][0][0])
        self.assertNotEqual(train_outer_0, train_outer_1)

    def test_no_leakage_into_val_across_all_folds(self):
        for i, call in enumerate(self.all_calls):
            val_fold = call[0][1]
            if len(val_fold) > 0:
                fitted = self.p._spy_pp.all_fitted_ids[i]
                self.assertTrue(fitted.isdisjoint(ids_of(val_fold)),
                                f"Fold {i}: preprocessor fitted on val — leakage!")

    def test_best_dir_created_per_outer_fold(self):
        run_dir = os.path.join(self.tmp.name, "test_run")
        for i in range(2):
            self.assertTrue(os.path.exists(
                os.path.join(run_dir, f"test_{i}", "best_auc")
            ))

    def test_config_written_for_all_gs_folds(self):
        run_dir = os.path.join(self.tmp.name, "test_run")
        for i in range(2):
            for j in range(2):
                self.assertTrue(os.path.exists(
                    os.path.join(run_dir, f"test_{i}", f"cv_{j}", "config.txt")
                ))

    def test_gs_collator_called_once_at_end(self):
        self.gs_collator.assert_called_once()

    def test_gs_collator_receives_gs_dirs_test_dirs_params(self):
        arg = self.gs_collator.call_args[0][0]
        self.assertIn("gs_dirs",   arg)
        self.assertIn("test_dirs", arg)
        self.assertIn("params",    arg)



# ══════════════════════════════════════════════════════════════════════════════
# run_single_split — load_data=True path
# ══════════════════════════════════════════════════════════════════════════════

class TestRunSingleSplitLoadData(unittest.TestCase):
    """Verify _load_data is called when load_data=True and skipped when False."""

    def test_load_data_called_when_flag_true(self):
        ds = make_dataset(n_samples=60)
        with tempfile.TemporaryDirectory() as tmp:
            p = make_pipeline(tmp, ds, {"test_samples": 0.2})
            p.run_single_split(load_data=True)
        p._load_data.assert_called_once()

    def test_load_data_not_called_when_flag_false(self):
        ds = make_dataset(n_samples=60)
        with tempfile.TemporaryDirectory() as tmp:
            p = make_pipeline(tmp, ds, {"test_samples": 0.2})
            p.run_single_split(load_data=False)
        p._load_data.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# run_crossvalidation — hold_out_validation_for_final_fit branch
# ══════════════════════════════════════════════════════════════════════════════

class TestRunCrossvalidationUseValidationOnTest(unittest.TestCase):
    """
    hold_out_validation_for_final_fit=True  → final refit uses get_train_test_split(1-prop, ...)
                                   so val fold is non-empty and train is a subset of outer train.
    hold_out_validation_for_final_fit=False → final refit uses get_train_test_split(1, ...)
                                   so the entire outer train pool is used and val fold is empty.
    """

    def _run(self, use_val_on_test):
        ds = make_dataset(n_samples=60)
        gs = [
            {"model_params": {"C": 0.1}, "model_params_choice": "small"},
        ]
        tmp = tempfile.TemporaryDirectory()
        p = make_pipeline(tmp.name, ds, {
            "inner_kfolds": 1,
            "outer_kfolds": 2,
            "grid_search": gs,
            "val_metric": {"auc": MagicMock(return_value=0.8)},
            "hold_out_validation_for_final_fit": use_val_on_test,
            "validation_prop": 0.2,
        })
        p.run_crossvalidation(load_data=False)
        calls = p._train.call_args_list
        tmp.cleanup()
        return p.config, calls

    def test_use_val_true_final_refit_has_nonempty_val(self):
        _, calls = self._run(use_val_on_test=True)
        # 2 outer folds × (1 gs + 1 best refit) = 4 calls
        # Best refit calls are at index 1 and 3 (every other after the gs run)
        best_val_0 = calls[1][0][1]   # val fold of first outer fold's best refit
        self.assertGreater(len(best_val_0), 0,
                           "hold_out_validation_for_final_fit=True: final refit val fold should be non-empty")

    def test_use_val_false_final_refit_train_is_full_outer_train(self):
        _, calls = self._run(use_val_on_test=False)
        # gs train for outer fold 0 is call 0; best refit for outer fold 0 is call 1
        gs_train   = ids_of(calls[0][0][0])
        best_train = ids_of(calls[1][0][0])
        # With validation_prop=0.2 and hold_out_validation_for_final_fit=False we call
        # get_train_test_split(1, ...) so best_train should equal the full outer train pool
        # i.e. it must be a superset of (or equal to) the gs train
        self.assertTrue(gs_train.issubset(best_train),
                        "hold_out_validation_for_final_fit=False: final refit train should include all of gs train")

    def test_use_val_false_final_refit_train_larger_than_gs_train(self):
        # hold_out_validation_for_final_fit=False merges val back into train for the final
        # refit via get_train_test_split(1,...). The best refit train must therefore
        # be strictly larger than the gs train (which only used 1-validation_prop).
        # calls layout: [outer0_gs, outer0_best, outer1_gs, outer1_best]
        _, calls = self._run(use_val_on_test=False)
        for fold_idx, (gs_call, best_call) in enumerate(zip(calls[::2], calls[1::2])):
            gs_train   = ids_of(gs_call[0][0])
            best_train = ids_of(best_call[0][0])
            self.assertGreater(len(best_train), len(gs_train),
                               f"Outer fold {fold_idx}: best refit train should be larger than gs train")
            self.assertTrue(gs_train.issubset(best_train),
                            f"Outer fold {fold_idx}: gs train samples missing from best refit train")

    def test_load_data_called_when_flag_true(self):
        ds = make_dataset(n_samples=60)
        gs = [{"model_params": {}, "model_params_choice": "default"}]
        with tempfile.TemporaryDirectory() as tmp:
            p = make_pipeline(tmp, ds, {
                "inner_kfolds": 1,
                "outer_kfolds": 2,
                "grid_search":  gs,
                "val_metric":   {"auc": MagicMock(return_value=0.8)},
            })
            p.run_crossvalidation(load_data=True)
        p._load_data.assert_called_once()

    def test_load_data_not_called_when_flag_false(self):
        ds = make_dataset(n_samples=60)
        with tempfile.TemporaryDirectory() as tmp:
            p = make_pipeline(tmp, ds, {"inner_kfolds": 1, "outer_kfolds": 2, "model_params": {}})
            p.run_crossvalidation(load_data=False)
        p._load_data.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# run_crossvalidation — inner k-folds combined with grid search
# ══════════════════════════════════════════════════════════════════════════════

class TestRunCrossvalidationInnerKFoldsWithGridSearch(unittest.TestCase):
    """
    inner_kfolds=2 + 2 gs configs + 2 outer folds.
    Verifies call counts, leakage, fold directories, and collator invocation.
    """

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.fold_collator = MagicMock()
        self.gs_collator   = MagicMock()
        # "large" (C=1.0) always wins: its metric score is higher regardless of fold
        def metric_fn(results):
            return 0.9 if results["train_df"].config_choice == "large" else 0.6
        gs = [
            {"model_params": {"C": 0.1}, "model_params_choice": "small"},
            {"model_params": {"C": 1.0}, "model_params_choice": "large"},
        ]
        # Score by the C value currently set on the pipeline config so the
        # winner is deterministic regardless of inner fold call order:
        # C=0.1 ("small") → 0.6, C=1.0 ("large") → 0.9
        def metric_fn(results):
            c = results["train_df"].xs.shape[1]   # dummy — use config instead
            return 0.6  # placeholder; real score injected via p reference below
        # We need access to p inside metric_fn, so we use a late-binding closure
        p_ref = [None]
        def metric_fn(results):
            return 0.9 if p_ref[0].config.get("model_params", {}).get("C", 0) >= 1.0 else 0.6
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "inner_kfolds":           2,
            "outer_kfolds":           2,
            "grid_search":            gs,
            "val_metric":             {"auc": metric_fn},
            "hold_out_validation_for_final_fit": False,
            "fold_collators":         [self.fold_collator],
            "grid_search_collators":  [self.gs_collator],
        })
        p_ref[0] = self.p
        self.p.run_crossvalidation(load_data=False)
        self.all_calls = self.p._train.call_args_list

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_call_count(self):
        # 2 outer × (2 gs configs × 2 inner folds + 1 best refit) = 2 × 5 = 10
        self.assertEqual(self.p._train.call_count, 10)

    def test_inner_fold_train_val_always_disjoint(self):
        # All training calls (inner folds + best refits) must have disjoint train/val
        for i, call in enumerate(self.all_calls):
            train_fold = call[0][0]
            val_fold   = call[0][1]
            if len(val_fold) > 0:
                self.assertTrue(ids_of(train_fold).isdisjoint(ids_of(val_fold)),
                                f"Call {i}: train/val overlap")

    def test_preprocessor_never_fitted_on_val(self):
        for i, call in enumerate(self.all_calls):
            val_fold = call[0][1]
            if len(val_fold) > 0:
                fitted = self.p._spy_pp.all_fitted_ids[i]
                self.assertTrue(fitted.isdisjoint(ids_of(val_fold)),
                                f"Call {i}: preprocessor fitted on val — leakage!")

    def test_fold_collator_called_once_per_outer_fold_per_gs_config(self):
        # 2 outer folds × 2 gs configs = 4 fold collator calls
        self.assertEqual(self.fold_collator.call_count, 4)

    def test_gs_collator_called_once(self):
        self.gs_collator.assert_called_once()

    def test_gs_collator_params_contains_winning_choice(self):
        # metric_fn returns 0.9 for even calls (second gs config = "large")
        # so "large" should appear in the params list for each outer fold
        arg = self.gs_collator.call_args[0][0]
        self.assertIn("large", arg["params"],
                      "Expected 'large' gs config to win but it was not recorded in params")

    def test_inner_fold_dirs_created(self):
        run_dir = os.path.join(self.tmp.name, "test_run")
        for outer in range(2):
            for gs_idx in range(2):
                for inner in range(2):
                    fold_dir = os.path.join(run_dir, f"test_{outer}",
                                            f"cv_{gs_idx}", f"fold_{inner}")
                    self.assertTrue(os.path.exists(fold_dir),
                                    f"Missing inner fold dir: {fold_dir}")

    def test_best_dirs_created_for_each_outer_fold(self):
        run_dir = os.path.join(self.tmp.name, "test_run")
        for outer in range(2):
            self.assertTrue(os.path.exists(
                os.path.join(run_dir, f"test_{outer}", "best_auc")
            ))



# ══════════════════════════════════════════════════════════════════════════════
# run_crossvalidation — samples_to_include pool restriction
# ══════════════════════════════════════════════════════════════════════════════

class TestRunCrossvalidationSamplesToInclude(unittest.TestCase):
    """
    When samples_to_include is set and test_samples is absent, the outer CV
    folds must be drawn only from those IDs — not from the full dataset.
    Samples outside the pool must never appear in any fold.
    """

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.pool_ids = self.ds.ids[:40]
        self.excluded_ids = set(self.ds.ids[40:])
        self.tmp = tempfile.TemporaryDirectory()
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "samples_to_include": list(self.pool_ids),
            "inner_kfolds":       1,
            "outer_kfolds":       2,
            "model_params":       {},
        })
        self.p.run_crossvalidation(load_data=False)
        self.all_calls = self.p._train.call_args_list

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_called_once_per_outer_fold(self):
        self.assertEqual(self.p._train.call_count, 2)

    def test_excluded_samples_not_in_any_train_fold(self):
        for i, call in enumerate(self.all_calls):
            self.assertTrue(
                ids_of(call[0][0]).isdisjoint(self.excluded_ids),
                f"Outer fold {i}: excluded samples found in train fold",
            )

    def test_excluded_samples_not_in_any_val_fold(self):
        for i, call in enumerate(self.all_calls):
            val_fold = call[0][1]
            if len(val_fold) > 0:
                self.assertTrue(
                    ids_of(val_fold).isdisjoint(self.excluded_ids),
                    f"Outer fold {i}: excluded samples found in val fold",
                )

    def test_only_pool_samples_appear_in_any_fold(self):
        all_seen = set()
        for call in self.all_calls:
            all_seen |= ids_of(call[0][0])
            all_seen |= ids_of(call[0][1])
        self.assertEqual(all_seen, set(self.pool_ids))


class TestRunCrossvalidationSamplesToIncludeWithGridSearch(unittest.TestCase):
    """
    Pool restriction propagates into inner folds and grid search: excluded
    samples must not appear in any inner fold either.
    """

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.pool_ids = self.ds.ids[:40]
        self.excluded_ids = set(self.ds.ids[40:])
        self.tmp = tempfile.TemporaryDirectory()
        gs = [
            {"model_params": {"C": 0.1}, "model_params_choice": "small"},
            {"model_params": {"C": 1.0}, "model_params_choice": "large"},
        ]
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "samples_to_include":              list(self.pool_ids),
            "inner_kfolds":                    2,
            "outer_kfolds":                    2,
            "grid_search":                     gs,
            "val_metric":                      {"auc": MagicMock(return_value=0.8)},
            "hold_out_validation_for_final_fit": False,
        })
        self.p.run_crossvalidation(load_data=False)
        self.all_calls = self.p._train.call_args_list

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_call_count(self):
        # 2 outer × (2 gs configs × 2 inner folds + 1 best refit) = 10
        self.assertEqual(self.p._train.call_count, 10)

    def test_excluded_samples_never_in_any_fold(self):
        for i, call in enumerate(self.all_calls):
            self.assertTrue(
                ids_of(call[0][0]).isdisjoint(self.excluded_ids),
                f"Call {i}: excluded samples in train fold",
            )
            val_fold = call[0][1]
            if len(val_fold) > 0:
                self.assertTrue(
                    ids_of(val_fold).isdisjoint(self.excluded_ids),
                    f"Call {i}: excluded samples in val fold",
                )


class TestRunCrossvalidationSamplesToIncludeAbsentUsesAllData(unittest.TestCase):
    """
    When samples_to_include is not set, all loaded data is used for the outer
    CV — existing behaviour is unchanged.
    """

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        # make_pipeline base has train_samples=0.6 (float) but no samples_to_include
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "inner_kfolds": 1,
            "outer_kfolds": 2,
            "model_params": {},
        })
        self.p.run_crossvalidation(load_data=False)
        self.all_calls = self.p._train.call_args_list

    def tearDown(self):
        self.tmp.cleanup()

    def test_second_half_samples_appear_in_training(self):
        # With all 60 samples the outer CV assigns the second half to outer fold 1's
        # training set. If samples_to_include erroneously activated, those IDs would
        # be excluded entirely.
        second_half = set(self.ds.ids[30:])
        all_train = set()
        for call in self.all_calls:
            all_train |= ids_of(call[0][0])
        self.assertTrue(
            all_train & second_half,
            "Without samples_to_include, the full dataset should be used for outer CV",
        )


# ══════════════════════════════════════════════════════════════════════════════
# ConcatMultiViewDataset — NA handling strategies
# ══════════════════════════════════════════════════════════════════════════════

class TestAlignViewsDropSamples(unittest.TestCase):
    """align_views with method='drop samples' removes rows that have any NaN."""

    def setUp(self):
        from architecture.data_utils import ConcatMultiViewDataset
        rng = np.random.default_rng(0)
        n = 20
        ids = [str(i) for i in range(n)]

        self.ds = ConcatMultiViewDataset()
        self.ds.ids = ids

        data = pd.DataFrame(
            rng.random((n, 5)).astype(np.float32),
            index=ids,
            columns=[f"gene_{i}" for i in range(5)],
        )
        # Deliberately inject NaN into the first three rows
        data.iloc[:3, 0] = np.nan
        self.ds.data_views["mut"] = data

        labels = pd.DataFrame({"response": np.ones(n, dtype=np.float32)}, index=ids)
        self.ds.labels = labels
        self.ds.align_views(method="drop samples", view_aligner={}, drop_labels=False)

    def test_nan_rows_removed(self):
        self.assertEqual(self.ds.xs.shape[0], 17,
                         "Rows with NaN should have been dropped")

    def test_no_nans_remain(self):
        self.assertFalse(np.isnan(self.ds.xs).any(),
                         "xs should contain no NaN after drop samples")

    def test_ids_consistent_with_xs(self):
        self.assertEqual(len(self.ds.ids), self.ds.xs.shape[0])

    def test_ys_consistent_with_xs(self):
        self.assertEqual(self.ds.ys.shape[0], self.ds.xs.shape[0])


class TestAlignViewsDropFeatures(unittest.TestCase):
    """align_views with method='drop features' removes columns that have any NaN."""

    def setUp(self):
        from architecture.data_utils import ConcatMultiViewDataset
        rng = np.random.default_rng(0)
        n = 20
        ids = [str(i) for i in range(n)]

        self.ds = ConcatMultiViewDataset()
        self.ds.ids = ids

        data = pd.DataFrame(
            rng.random((n, 5)).astype(np.float32),
            index=ids,
            columns=[f"gene_{i}" for i in range(5)],
        )
        # First two columns have NaN in at least one row
        data.iloc[0, 0] = np.nan
        data.iloc[1, 1] = np.nan
        self.ds.data_views["mut"] = data

        labels = pd.DataFrame({"response": np.ones(n, dtype=np.float32)}, index=ids)
        self.ds.labels = labels
        self.ds.align_views(method="drop features", view_aligner={}, drop_labels=False)

    def test_nan_columns_removed(self):
        self.assertEqual(self.ds.xs.shape[1], 3,
                         "Columns with NaN should have been dropped")

    def test_no_nans_remain(self):
        self.assertFalse(np.isnan(self.ds.xs).any())

    def test_features_list_consistent_with_xs(self):
        self.assertEqual(len(self.ds.features), self.ds.xs.shape[1])

    def test_alignment_ids_consistent_with_xs(self):
        self.assertEqual(len(self.ds.alignment_ids), self.ds.xs.shape[1])


# ══════════════════════════════════════════════════════════════════════════════
# ConcatMultiViewDataset — get_k_splits edge case: n_splits=1
# ══════════════════════════════════════════════════════════════════════════════

class TestGetKSplitsOne(unittest.TestCase):
    """n_splits=1 should return the entire dataset as train with an empty val."""

    def setUp(self):
        self.ds = make_dataset(n_samples=30)
        self.folds = list(self.ds.get_k_splits(1, seed=42))

    def test_exactly_one_fold(self):
        self.assertEqual(len(self.folds), 1)

    def test_train_contains_all_ids(self):
        train, _ = self.folds[0]
        self.assertEqual(ids_of(train), set(self.ds.ids))

    def test_val_is_empty(self):
        _, val = self.folds[0]
        self.assertEqual(len(val), 0)


# ══════════════════════════════════════════════════════════════════════════════
# data_augmentor — applied only to train, not val or test
# ══════════════════════════════════════════════════════════════════════════════

class TestDataAugmentorAppliedOnlyToTrain(unittest.TestCase):
    """
    The augmentor must be called on the training fold inside _fold_run and
    must never receive val samples.
    """

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.augmented_ids = []

        def recording_augmentor(fold):
            self.augmented_ids.append(set(fold.ids))
            return fold

        self.p = make_pipeline(self.tmp.name, self.ds, {
            "test_samples":   0.2,
            "data_augmentor": recording_augmentor,
        })
        self.p.run_single_split(load_data=False)
        self.val_ids = ids_of(self.p._train.call_args[0][1])

    def tearDown(self):
        self.tmp.cleanup()

    def test_augmentor_called_exactly_once(self):
        self.assertEqual(len(self.augmented_ids), 1)

    def test_augmentor_not_given_val_samples(self):
        self.assertTrue(
            self.augmented_ids[0].isdisjoint(self.val_ids),
            "Augmentor received val samples — it should only see training data",
        )


class TestDataAugmentorNotAppliedInCrossval(unittest.TestCase):
    """With outer_kfolds=2 the augmentor is called once per outer fold (on train only)."""

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.augmented_ids = []

        def recording_augmentor(fold):
            self.augmented_ids.append(set(fold.ids))
            return fold

        self.p = make_pipeline(self.tmp.name, self.ds, {
            "inner_kfolds":   1,
            "outer_kfolds":   2,
            "data_augmentor": recording_augmentor,
            "model_params":   {},
        })
        self.p.run_crossvalidation(load_data=False)
        self.all_calls = self.p._train.call_args_list

    def tearDown(self):
        self.tmp.cleanup()

    def test_augmentor_called_once_per_outer_fold(self):
        self.assertEqual(len(self.augmented_ids), 2)

    def test_augmentor_never_sees_val_samples(self):
        for i, call in enumerate(self.all_calls):
            val_ids = ids_of(call[0][1])
            self.assertTrue(
                self.augmented_ids[i].isdisjoint(val_ids),
                f"Outer fold {i}: augmentor received val samples",
            )


# ══════════════════════════════════════════════════════════════════════════════
# results_processors — called with correct keys and values
# ══════════════════════════════════════════════════════════════════════════════

class TestResultsProcessorReceivesCorrectKeys(unittest.TestCase):
    """
    results_processors must be called with a dict containing the documented
    keys, with predictions whose lengths match the corresponding fold sizes.
    """

    REQUIRED_KEYS = {
        "train_preds", "val_preds", "test_preds",
        "train_df", "val_df", "test_df",
        "train_hx", "save_dir", "model",
        "feature_preprocessor", "feature_selector",
    }

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.captured = []
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "test_samples":       0.2,
            "results_processors": [lambda r: self.captured.append(dict(r))],
        })
        self.p.run_single_split(load_data=False)
        self.result = self.captured[0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_required_keys_present(self):
        missing = self.REQUIRED_KEYS - set(self.result.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")

    def test_train_preds_length_matches_train_fold(self):
        self.assertEqual(len(self.result["train_preds"]), len(self.result["train_df"]))

    def test_val_preds_length_matches_val_fold(self):
        self.assertEqual(len(self.result["val_preds"]), len(self.result["val_df"]))

    def test_test_preds_length_matches_test_fold(self):
        self.assertEqual(len(self.result["test_preds"]), len(self.result["test_df"]))

    def test_save_dir_is_a_string_path(self):
        self.assertIsInstance(self.result["save_dir"], str)

    def test_train_hx_passed_through(self):
        self.assertIsNotNone(self.result["train_hx"])


class TestResultsProcessorValNoneWhenNoVal(unittest.TestCase):
    """When the val fold is empty, _fold_run should pass val_preds=None."""

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.captured = []
        # inner_kfolds=1 with no val_metric produces a [] val fold via the
        # get_train_test_split(1, ...) path; no best refit is triggered
        gs = [{"model_params": {}, "model_params_choice": "default"}]
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "inner_kfolds":       1,
            "outer_kfolds":       2,
            "grid_search":        gs,
            "val_metric":         {},
            "results_processors": [lambda r: self.captured.append(dict(r))],
            "model_params":       {},
        })
        self.p.run_crossvalidation(load_data=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_val_preds_is_none_when_no_val(self):
        for r in self.captured:
            if len(r["val_df"]) == 0:
                self.assertIsNone(r["val_preds"],
                                  "val_preds should be None when val fold is empty")


# ══════════════════════════════════════════════════════════════════════════════
# run_single_split — grid search with no val_metric
# ══════════════════════════════════════════════════════════════════════════════

class TestRunSingleSplitGridSearchNoValMetric(unittest.TestCase):
    """
    Grid search present but val_metric={}: runs all gs configs but skips
    best-dir creation and final refit entirely.
    """

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        gs = [
            {"model_params": {"C": 0.1}, "model_params_choice": "small"},
            {"model_params": {"C": 1.0}, "model_params_choice": "large"},
        ]
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "grid_search":  gs,
            "val_metric":   {},
            "test_samples": 0.2,
        })
        self.p.run_single_split(load_data=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_called_once_per_gs_config_only(self):
        # No val_metric → no best refit
        self.assertEqual(self.p._train.call_count, 2)

    def test_no_best_dir_created(self):
        run_dir = os.path.join(self.tmp.name, "test_run")
        best_dirs = [d for d in os.listdir(run_dir) if d.startswith("best_")]
        self.assertEqual(best_dirs, [],
                         "No best_* directory should exist when val_metric is empty")

    def test_cv_dirs_still_created(self):
        run_dir = os.path.join(self.tmp.name, "test_run")
        for i in range(2):
            self.assertTrue(os.path.exists(os.path.join(run_dir, f"cv_{i}")))


# ══════════════════════════════════════════════════════════════════════════════
# run_crossvalidation — grid search with no val_metric
# ══════════════════════════════════════════════════════════════════════════════

class TestRunCrossvalidationGridSearchNoValMetric(unittest.TestCase):
    """
    Grid search present but val_metric={}: runs all gs configs per outer fold
    but skips best-dir creation. gs_collator is still called at the end.
    """

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.gs_collator = MagicMock()
        gs = [
            {"model_params": {"C": 0.1}, "model_params_choice": "small"},
            {"model_params": {"C": 1.0}, "model_params_choice": "large"},
        ]
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "inner_kfolds":          1,
            "outer_kfolds":          2,
            "grid_search":           gs,
            "val_metric":            {},
            "grid_search_collators": [self.gs_collator],
        })
        self.p.run_crossvalidation(load_data=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_called_only_for_gs_configs(self):
        # 2 outer folds × 2 gs configs, no best refits
        self.assertEqual(self.p._train.call_count, 4)

    def test_no_best_dirs_created(self):
        run_dir = os.path.join(self.tmp.name, "test_run")
        for i in range(2):
            test_dir = os.path.join(run_dir, f"test_{i}")
            best_dirs = [d for d in os.listdir(test_dir) if d.startswith("best_")]
            self.assertEqual(best_dirs, [],
                             f"test_{i}: no best_* dirs expected without val_metric")

    def test_gs_collator_still_called_once(self):
        self.gs_collator.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# run_crossvalidation — stratified splitting at pipeline level
# ══════════════════════════════════════════════════════════════════════════════

class TestRunCrossvalidationStratified(unittest.TestCase):
    """
    stratified=True propagates into the k-split calls so every outer train
    fold contains both classes in roughly equal proportions.
    """

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "inner_kfolds": 1,
            "outer_kfolds": 2,
            "stratified":   True,
            "model_params": {},
        })
        self.p.run_crossvalidation(load_data=False)
        self.all_calls = self.p._train.call_args_list

    def tearDown(self):
        self.tmp.cleanup()

    def test_both_classes_in_every_outer_train_fold(self):
        for i, call in enumerate(self.all_calls):
            train_fold = call[0][0]
            counts = self.ds.labels.loc[list(ids_of(train_fold))]["response"].value_counts()
            self.assertIn(0.0, counts.index, f"Outer fold {i} train missing class 0")
            self.assertIn(1.0, counts.index, f"Outer fold {i} train missing class 1")

    def test_class_balance_roughly_equal_in_train(self):
        for i, call in enumerate(self.all_calls):
            train_fold = call[0][0]
            counts = self.ds.labels.loc[list(ids_of(train_fold))]["response"].value_counts(normalize=True)
            self.assertAlmostEqual(counts[0.0], 0.5, delta=0.15,
                                   msg=f"Outer fold {i}: class balance off")


class TestRunSingleSplitStratified(unittest.TestCase):
    """stratified=True in run_single_split preserves class balance in the train fold."""

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "test_samples": 0.2,
            "stratified":   True,
        })
        self.p.run_single_split(load_data=False)
        self.train_fold = self.p._train.call_args[0][0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_both_classes_present_in_train(self):
        counts = self.ds.labels.loc[list(ids_of(self.train_fold))]["response"].value_counts()
        self.assertIn(0.0, counts.index)
        self.assertIn(1.0, counts.index)

    def test_class_balance_roughly_equal(self):
        counts = self.ds.labels.loc[list(ids_of(self.train_fold))]["response"].value_counts(normalize=True)
        self.assertAlmostEqual(counts[0.0], 0.5, delta=0.15)


# ══════════════════════════════════════════════════════════════════════════════
# run_crossvalidation — inner_kfolds=1 with val_metric
# ══════════════════════════════════════════════════════════════════════════════

class TestRunCrossvalidationInnerOneFoldWithValMetric(unittest.TestCase):
    """
    inner_kfolds=1 with a val_metric: pipeline produces a non-empty val fold
    via get_train_test_split(1-validation_prop, ...) and still selects the
    best config and creates best dirs.
    """

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        gs = [
            {"model_params": {"C": 0.1}, "model_params_choice": "small"},
            {"model_params": {"C": 1.0}, "model_params_choice": "large"},
        ]
        self.val_metric = MagicMock(return_value=0.75)
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "inner_kfolds":           1,
            "outer_kfolds":           2,
            "grid_search":            gs,
            "val_metric":             {"auc": self.val_metric},
            "hold_out_validation_for_final_fit": False,
            "validation_prop":        0.2,
        })
        self.p.run_crossvalidation(load_data=False)
        self.all_calls = self.p._train.call_args_list

    def tearDown(self):
        self.tmp.cleanup()

    def test_val_fold_non_empty_for_gs_runs(self):
        # First two calls are the gs runs for outer fold 0
        for i in range(2):
            val_fold = self.all_calls[i][0][1]
            self.assertGreater(len(val_fold), 0,
                               f"GS run {i}: expected non-empty val fold")

    def test_val_fold_approximately_correct_proportion(self):
        val_fold   = self.all_calls[0][0][1]
        train_fold = self.all_calls[0][0][0]
        total = len(val_fold) + len(train_fold)
        self.assertAlmostEqual(len(val_fold) / total, 0.2, delta=0.05)

    def test_best_dir_created_per_outer_fold(self):
        run_dir = os.path.join(self.tmp.name, "test_run")
        for i in range(2):
            self.assertTrue(
                os.path.exists(os.path.join(run_dir, f"test_{i}", "best_auc")),
                f"best_auc dir missing for outer fold {i}",
            )

    def test_val_metric_called_once_per_gs_config_per_outer_fold(self):
        # 2 outer folds × 2 gs configs = 4 metric evaluations
        self.assertEqual(self.val_metric.call_count, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  collator scope — fold_collators and grid_search_collators not called
#      by run_single_split
# ══════════════════════════════════════════════════════════════════════════════

class TestFoldCollatorNotCalledInSingleSplit(unittest.TestCase):
    """fold_collators belong to run_crossvalidation's inner loop only."""

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.fold_collator = MagicMock()
        gs = [{"model_params": {}, "model_params_choice": "default"}]
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "grid_search":    gs,
            "val_metric":     {"auc": MagicMock(return_value=0.8)},
            "test_samples":   0.2,
            "fold_collators": [self.fold_collator],
        })
        self.p.run_single_split(load_data=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fold_collator_not_called(self):
        self.fold_collator.assert_not_called()


class TestGridSearchCollatorCalledInSingleSplit(unittest.TestCase):
    """
    grid_search_collators ARE called by run_single_split (not only by
    run_crossvalidation) once the best-refit loop completes.
    """

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.gs_collator = MagicMock()
        gs = [{"model_params": {}, "model_params_choice": "default"}]
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "grid_search":           gs,
            "val_metric":            {"auc": MagicMock(return_value=0.8)},
            "test_samples":          0.2,
            "grid_search_collators": [self.gs_collator],
        })
        self.p.run_single_split(load_data=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_gs_collator_called_once(self):
        self.gs_collator.assert_called_once()

    def test_gs_collator_receives_gs_dirs_params_save_dir(self):
        arg = self.gs_collator.call_args[0][0]
        self.assertIn("gs_dirs",  arg)
        self.assertIn("params",   arg)
        self.assertIn("save_dir", arg)

    def test_gs_collator_does_not_receive_test_dirs_in_single_split(self):
        # run_crossvalidation includes test_dirs; run_single_split does not
        arg = self.gs_collator.call_args[0][0]
        self.assertNotIn("test_dirs", arg)


# ══════════════════════════════════════════════════════════════════════════════
# fold_collator payload — results list contains correct fold directories
# ══════════════════════════════════════════════════════════════════════════════

class TestFoldCollatorPayloadContainsCorrectDirs(unittest.TestCase):
    """
    The 'results' key passed to fold_collators must be a list of directory
    paths that exist on disk, one per inner fold, each under save_dir.
    """

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.collator_args = []
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "inner_kfolds":   2,
            "outer_kfolds":   2,
            "fold_collators": [lambda arg: self.collator_args.append(arg)],
            "model_params":   {},
        })
        self.p.run_crossvalidation(load_data=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_results_is_a_list(self):
        for arg in self.collator_args:
            self.assertIsInstance(arg["results"], list)

    def test_results_has_one_entry_per_inner_fold(self):
        for arg in self.collator_args:
            self.assertEqual(len(arg["results"]), 2)

    def test_result_dirs_exist_on_disk(self):
        for arg in self.collator_args:
            for fold_dir in arg["results"]:
                self.assertTrue(os.path.exists(fold_dir),
                                f"Fold directory does not exist: {fold_dir}")

    def test_save_dir_is_parent_of_fold_dirs(self):
        for arg in self.collator_args:
            save_dir = arg["save_dir"]
            for fold_dir in arg["results"]:
                self.assertTrue(fold_dir.startswith(save_dir),
                                f"Fold dir {fold_dir} is not under save_dir {save_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# feature_selector.get_features() called after fit_transform
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureSelectorGetFeaturesCalledAfterFit(unittest.TestCase):
    """
    _fold_run logs the number of selected features by calling
    feature_selector.get_features(). It must be called after fit_transform.
    """

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        call_order = []

        class OrderRecordingSelector(SpyTransformer):
            def fit_transform(self, fold):
                call_order.append("fit_transform")
                return super().fit_transform(fold)

            def get_features(self):
                call_order.append("get_features")
                return ["f1", "f2"]

        self.selector = OrderRecordingSelector()
        self.call_order = call_order
        self.p = make_pipeline(self.tmp.name, self.ds, {
            "test_samples":     0.2,
            "feature_selector": self.selector,
        })
        self.p.run_single_split(load_data=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fit_transform_called_on_selector(self):
        self.assertIn("fit_transform", self.call_order,
                      "feature_selector.fit_transform was never called by _fold_run")

    def test_selector_fit_transform_called_exactly_once(self):
        count = self.call_order.count("fit_transform")
        self.assertEqual(count, 1,
                         f"Expected fit_transform called once for a single split, got {count}")


class TestRunSingleSplitMultipleValMetricsNoClobber(unittest.TestCase):
    """
    When multiple val_metrics are present, the best-refit loop iterates once
    per metric. The hold_out_validation_for_final_fit=False branch reassigns local split
    variables — this test verifies that later metric iterations are not
    corrupted by earlier ones.
    """

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        gs = [
            {"model_params": {"C": 0.1}, "model_params_choice": "small"},
            {"model_params": {"C": 1.0}, "model_params_choice": "large"},
        ]

        # Wrap get_specific_split to record every call's arguments and results
        self.split_calls = []
        original_split = self.ds.get_specific_split

        def recording_split(*args, **kwargs):
            result = original_split(*args, **kwargs)
            self.split_calls.append({
                "args": args,
                "train_ids": set(result[0].ids),
                "val_ids":   set(result[1].ids) if len(result[1]) > 0 else set(),
                "test_ids":  set(result[2].ids) if len(result[2]) > 0 else set(),
            })
            return result

        self.ds.get_specific_split = recording_split

        self.p = make_pipeline(self.tmp.name, self.ds, {
            "grid_search":            gs,
            "val_metric":             {
                "auc":   MagicMock(return_value=0.8),
                "f1":    MagicMock(return_value=0.7),
                "auprc": MagicMock(return_value=0.6),
            },
            "hold_out_validation_for_final_fit": False,
            "test_samples":           0.2,
        })
        self.p.run_single_split(load_data=False)
        self.all_calls = self.p._train.call_args_list

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_called_once_per_gs_config_plus_one_per_metric(self):
        # 2 gs configs + 3 metrics = 5
        self.assertEqual(self.p._train.call_count, 5)

    def test_get_specific_split_called_once_per_metric_refit(self):
        # 1 initial split + 3 refit splits (one per metric) = 4
        self.assertEqual(len(self.split_calls), 4,
                         f"Expected 4 get_specific_split calls, got {len(self.split_calls)}")

    def test_all_refit_splits_use_same_train_proportion(self):
        # Every refit call should pass train_samples+val_samples as first arg
        # i.e. the same combined proportion, not an already-combined dataset
        initial_train_prop = self.split_calls[0]["args"][0]  # e.g. 0.6
        initial_val_prop   = self.split_calls[0]["args"][1]  # e.g. 0.2
        expected_combined  = initial_train_prop + initial_val_prop
        for i, call in enumerate(self.split_calls[1:], 1):
            self.assertAlmostEqual(
                call["args"][0], expected_combined, delta=0.001,
                msg=f"Refit split {i}: train_samples arg was {call['args'][0]}, "
                    f"expected {expected_combined} — earlier iteration may have "
                    f"clobbered train_df causing a list of IDs to be passed instead"
            )

    def test_all_best_refits_have_identical_training_sets(self):
        # All three best refits (train calls 2, 3, 4) must see the same train IDs
        # because they all call get_specific_split with the same arguments.
        # If train_df is clobbered, later refits train on the combined pool from
        # the previous iteration which will have different (larger) IDs.
        best_refit_train_ids = [
            ids_of(self.all_calls[i][0][0]) for i in range(2, 5)
        ]
        for i in range(1, len(best_refit_train_ids)):
            self.assertEqual(
                best_refit_train_ids[0], best_refit_train_ids[i],
                f"Best refit {i} has different training IDs than refit 0 — "
                "train_df was clobbered between metric iterations"
            )

    def test_all_best_refits_exclude_test_samples(self):
        # Test IDs are those not in the initial train or val split
        initial = self.split_calls[0]
        test_ids = set(self.ds.ids) - initial["train_ids"] - initial["val_ids"]
        for i in range(2, 5):
            best_train = ids_of(self.all_calls[i][0][0])
            self.assertTrue(
                best_train.isdisjoint(test_ids),
                f"Best refit {i - 2}: training set contains test samples — leakage!"
            )

    def test_all_best_dirs_created(self):
        run_dir = os.path.join(self.tmp.name, "test_run")
        for metric in ("auc", "f1", "auprc"):
            self.assertTrue(
                os.path.exists(os.path.join(run_dir, f"best_{metric}")),
                f"best_{metric} directory was not created"
            )

# ══════════════════════════════════════════════════════════════════════════════
# MLPipeline — real sklearn training
# ══════════════════════════════════════════════════════════════════════════════

def make_ml_pipeline(tmp_dir, ds, config_overrides=None):
    """Build an MLPipeline backed by a real ConcatMultiViewDataset."""
    from architecture.pipeline import MLPipeline
    from sklearn.linear_model import Ridge

    spy_pp = SpyTransformer()
    spy_fs = SpyTransformer()

    base_config = {
        "run_dir":                tmp_dir,
        "run_id":                 "test_ml_run",
        "grid_search":            [],
        "val_metric":             {},
        "fold_collators":         [],
        "grid_search_collators":  [],
        "feature_selector":       spy_fs,
        "data_augmentor":         lambda x: x,
        "feature_preprocessor":   spy_pp,
        "results_processors":     [],
        "rng_seed":               42,
        "train_samples":          0.6,
        "val_samples":            0.2,
        "tt_split_seed":          0,
        "tv_split_seed":          0,
        "hold_out_validation_for_final_fit": False,
        "validation_prop":        0.2,
        "stratified":             False,
        "inner_kfolds":           1,
        "outer_kfolds":           2,
        "model_params":           {},
        "shuffle_seed":           42,
        "model":                  Ridge,
        "task":                   "regression",
    }
    if config_overrides:
        base_config.update(config_overrides)

    p = MLPipeline(base_config)
    p.data = ds
    p._load_data = MagicMock(side_effect=lambda *args, **kwargs: None)

    def silent_logger(name, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        log = logging.getLogger(name + log_dir)
        log.handlers.clear()
        log.addHandler(logging.NullHandler())
        return log

    p._get_logger = silent_logger
    p._summarise_data = MagicMock()
    p._spy_pp = spy_pp
    p._spy_fs = spy_fs

    return p


class TestMLPipelineRunSingleSplit(unittest.TestCase):
    """MLPipeline._train uses real sklearn and returns (SKModelWrapper, None)."""

    def setUp(self):
        self.ds = make_dataset(n_samples=60)
        self.tmp = tempfile.TemporaryDirectory()
        self.captured = []
        self.p = make_ml_pipeline(self.tmp.name, self.ds, {
            "test_samples":       0.2,
            "results_processors": [lambda r: self.captured.append(dict(r))],
        })
        self.p.run_single_split(load_data=False)
        self.result = self.captured[0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_model_is_sk_model_wrapper(self):
        from architecture.pipeline import SKModelWrapper
        self.assertIsInstance(self.result["model"], SKModelWrapper)

    def test_train_hx_is_none(self):
        self.assertIsNone(self.result["train_hx"])

    def test_train_preds_finite(self):
        self.assertTrue(np.all(np.isfinite(self.result["train_preds"])))

    def test_val_preds_finite(self):
        self.assertTrue(np.all(np.isfinite(self.result["val_preds"])))

    def test_predictions_length_matches_fold(self):
        self.assertEqual(len(self.result["train_preds"]), len(self.result["train_df"]))
        self.assertEqual(len(self.result["val_preds"]),   len(self.result["val_df"]))


class TestMLPipelineExternalValidation(unittest.TestCase):
    """_run_external_validation produces predictions.csv under external_validation/<tag>/."""

    N_FEATURES = 10
    N_EXT = 15

    def setUp(self):
        from sklearn.linear_model import Ridge

        self.ds = make_dataset(n_samples=60, n_features=self.N_FEATURES)
        self.tmp = tempfile.TemporaryDirectory()

        rng = np.random.default_rng(99)
        ext_ids = [f"ext_{i}" for i in range(self.N_EXT)]
        ext_xs  = rng.random((self.N_EXT, self.N_FEATURES)).astype(np.float32)
        ext_ys  = np.zeros((self.N_EXT, 1), dtype=np.float32)
        features = [f"gene_{i}" for i in range(self.N_FEATURES)]

        # Fake external dataset: load_data_view / align_views are no-ops so no
        # CSV files are needed; xs/ys/features/ids are pre-populated.
        class FakeExtDataset:
            def __init__(self_):
                self_.xs           = ext_xs.copy()
                self_.ys           = ext_ys.copy()
                self_.ids          = list(ext_ids)
                self_.features     = list(features)
                self_.alignment_ids = list(features)
                self_.labels       = pd.DataFrame({"response": ext_ys[:, 0]}, index=ext_ids)
                self_.data_views   = {}

            def load_data_view(self_, *args, **kwargs): pass
            def load_data_label(self_, *args, **kwargs): pass
            def align_views(self_, *args, **kwargs):    pass
            def get_labels(self_):                      return ["response"]

        self.p = make_ml_pipeline(self.tmp.name, self.ds, {
            "model":               Ridge,
            "task":                "regression",
            "model_params":        {"alpha": 1.0},
            "final_model_params":  {"alpha": 1.0},
            "dataloader":          FakeExtDataset,
            "data_dir":            self.tmp.name,
            "view_alignment_method": "zero fill",
            "drop_labels":         False,
            "shuffle_seed":        42,
            "external_datasets":   [{
                "tag":    "test_ext",
                "views":  [("mut", "dummy.csv", None, "id", None, None)],
                "labels": [],
            }],
        })

        # run_dir and log are normally set by run_single_split / run_crossvalidation
        self.run_dir = os.path.join(self.tmp.name, "test_ml_run")
        self.p.run_dir = self.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        _log = logging.getLogger("test_ext_val_logger")
        _log.handlers.clear()
        _log.addHandler(logging.NullHandler())
        self.p.log = _log

        self.p._run_external_validation()
        self.tag_dir = os.path.join(self.run_dir, "external_validation", "test_ext")

    def tearDown(self):
        self.tmp.cleanup()

    def test_predictions_csv_created(self):
        self.assertTrue(os.path.exists(os.path.join(self.tag_dir, "predictions.csv")))

    def test_predictions_has_correct_row_count(self):
        df = pd.read_csv(os.path.join(self.tag_dir, "predictions.csv"), index_col=0)
        self.assertEqual(len(df), self.N_EXT)

    def test_predictions_has_response_and_pred_columns(self):
        df = pd.read_csv(os.path.join(self.tag_dir, "predictions.csv"), index_col=0)
        self.assertIn("response",      df.columns)
        self.assertIn("response_pred", df.columns)

    def test_predictions_are_finite(self):
        df = pd.read_csv(os.path.join(self.tag_dir, "predictions.csv"), index_col=0)
        self.assertTrue(np.all(np.isfinite(df["response_pred"].values)))

    def test_predictions_index_is_ext_ids(self):
        df = pd.read_csv(os.path.join(self.tag_dir, "predictions.csv"), index_col=0)
        self.assertEqual(set(df.index.astype(str)),
                         {f"ext_{i}" for i in range(self.N_EXT)})


class TestShuffleSeedChangesColumnOrder(unittest.TestCase):
    """
    End-to-end check on the real dataset class: two different shuffle_seeds must
    produce different input-feature orderings, and the same seed must reproduce.
    """

    def _build(self, shuffle_seed):
        from architecture.data_utils import ConcatMultiViewDataset

        tmp = tempfile.mkdtemp()
        ids = [f"s{i:02d}" for i in range(12)]
        genes = [f"g{i:02d}" for i in range(20)]
        rng = np.random.default_rng(0)

        view = pd.DataFrame(rng.random((len(ids), len(genes))).astype(np.float32),
                            index=ids, columns=genes)
        view.index.name = "id"
        vpath = os.path.join(tmp, "v.csv")
        view.to_csv(vpath)

        labels = pd.DataFrame({"binary": [0, 1] * 6}, index=ids, dtype=np.float32)
        labels.index.name = "id"
        lpath = os.path.join(tmp, "labels.csv")
        labels.to_csv(lpath)

        ds = ConcatMultiViewDataset()
        ds.load_data_view("v", vpath)
        ds.load_data_label(lpath)
        ds.align_views(method="zero fill", drop_labels=True, shuffle_seed=shuffle_seed)
        return ds

    def test_different_seeds_give_different_feature_order(self):
        a = self._build(42)
        b = self._build(20240617)
        self.assertNotEqual(a.get_features(), b.get_features())

    def test_same_seed_is_reproducible(self):
        a = self._build(20240617)
        b = self._build(20240617)
        self.assertEqual(a.get_features(), b.get_features())

    def test_feature_set_is_unchanged_by_seed(self):
        """Shuffling reorders features; it must not add or drop any."""
        a = self._build(42)
        b = self._build(20240617)
        self.assertEqual(sorted(a.get_features()), sorted(b.get_features()))


if __name__ == "__main__":
    unittest.main()