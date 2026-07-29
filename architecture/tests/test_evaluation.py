import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
import tempfile

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Minimal stubs — only what evaluation.py imports directly
# ---------------------------------------------------------------------------

# keras.models.Sequential
keras_stub = types.ModuleType("keras")
keras_models_stub = types.ModuleType("keras.models")

class _FakeSequential:
    def __init__(self):
        self.layers = []

keras_models_stub.Sequential = _FakeSequential
keras_stub.models = keras_models_stub
sys.modules["keras"] = keras_stub
sys.modules["keras.models"] = keras_models_stub

# pnet_model — stub just the two names evaluation.py imports
pnet_stub = types.ModuleType("pnet_model")
pnet_stub.get_layer_maps = MagicMock()
pnet_stub.PNetArchitectureGenerator = MagicMock()
sys.modules["pnet_model"] = pnet_stub

# architecture.coef_weights_utils — stub just the functions evaluation.py calls
mcw_stub = types.ModuleType("architecture.coef_weights_utils")
for _fn in [
    "get_gradient_weights",
    "get_weights_gradient_outcome",
    "get_gradient_weights_with_repeated_output",
    "get_permutation_weights",
    "get_weights_linear_model",
    "get_deep_explain_scores",
    "get_shap_scores",
    "get_skf_weights",
]:
    setattr(mcw_stub, _fn, MagicMock(return_value=np.ones(5)))

arch_stub = types.ModuleType("architecture")
arch_stub.coef_weights_utils = mcw_stub
arch_stub.pnet_model = pnet_stub
sys.modules["architecture"] = arch_stub
sys.modules["architecture.coef_weights_utils"] = mcw_stub
sys.modules["architecture.pnet_model"] = pnet_stub

import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Import module under test
# ---------------------------------------------------------------------------
import importlib.util
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
MODULE_PATH = TEST_DIR.parent / "evaluation.py"
if not MODULE_PATH.exists():
    raise ImportError(f"Could not find evaluation.py at {MODULE_PATH}")

_spec = importlib.util.spec_from_file_location("results_utils", str(MODULE_PATH))
ru = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ru)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_df(n_rows=10, n_labels=2, ids=None):
    mock = MagicMock()
    ys = np.random.rand(n_rows, n_labels)
    mock.ys = ys
    mock.ids = ids or [f"id_{i}" for i in range(n_rows)]
    mock.get_labels.return_value = [f"label_{i}" for i in range(n_labels)]
    mock.alignment_ids = [f"feat_{i}" for i in range(5)]
    mock.__len__ = lambda self: n_rows
    return mock


def _make_results(tmp_dir, n_rows=10, n_labels=2, include_val=True, include_test=True):
    train_df = _make_mock_df(n_rows, n_labels)
    val_df = _make_mock_df(n_rows, n_labels) if include_val else MagicMock(__len__=lambda s: 0)
    test_df = _make_mock_df(n_rows, n_labels) if include_test else MagicMock(__len__=lambda s: 0)

    preds = np.random.rand(n_rows, n_labels)

    return {
        "save_dir": tmp_dir,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_preds": preds,
        "val_preds": preds,
        "test_preds": preds,
        "model": MagicMock(),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCollateGridSearch(unittest.TestCase):

    def test_creates_results_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            gs_dir1 = os.path.join(tmp, "gs1")
            gs_dir2 = os.path.join(tmp, "gs2")
            os.makedirs(gs_dir1)
            os.makedirs(gs_dir2)

            pd.DataFrame({"metric": [0.8]}).to_csv(f"{gs_dir1}/summary_results.csv")
            pd.DataFrame({"metric": [0.9]}).to_csv(f"{gs_dir2}/summary_results.csv")

            results = {
                "save_dir": tmp,
                "params": [{"lr": 0.01}, {"lr": 0.001}],
                "gs_dirs": [gs_dir1, gs_dir2],
                "test_dirs": ["fold0", "fold1"],
            }

            ru.collate_grid_search(results)
            self.assertTrue(os.path.exists(f"{tmp}/results.csv"))

    def test_results_csv_contains_hyperparam_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            gs_dir = os.path.join(tmp, "gs1")
            os.makedirs(gs_dir)
            pd.DataFrame({"metric": [0.5]}).to_csv(f"{gs_dir}/summary_results.csv")

            results = {
                "save_dir": tmp,
                "params": [{"lr": 0.01}],
                "gs_dirs": [gs_dir],
                "test_dirs": ["fold0"],
            }
            ru.collate_grid_search(results)
            df = pd.read_csv(f"{tmp}/results.csv")
            self.assertIn("hyperparams", df.columns)
            self.assertIn("test_fold", df.columns)

    def test_multiple_gs_dirs_all_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = []
            for i in range(3):
                d = os.path.join(tmp, f"gs{i}")
                os.makedirs(d)
                pd.DataFrame({"metric": [i * 0.1]}).to_csv(f"{d}/summary_results.csv")
                dirs.append(d)

            results = {
                "save_dir": tmp,
                "params": [{"a": i} for i in range(3)],
                "gs_dirs": dirs,
                "test_dirs": [f"fold{i}" for i in range(3)],
            }
            ru.collate_grid_search(results)
            df = pd.read_csv(f"{tmp}/results.csv")
            self.assertEqual(len(df), 3)


class TestCollateFolds(unittest.TestCase):

    def test_creates_fold_summaries_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            fold_dirs = []
            for i in range(2):
                d = os.path.join(tmp, f"fold{i}")
                os.makedirs(d)
                pd.DataFrame({"metric": [i * 0.5]}).to_csv(f"{d}/summary_results.csv")
                fold_dirs.append(d)

            results = {"save_dir": tmp, "results": fold_dirs}
            ru.collate_folds(results)
            self.assertTrue(os.path.exists(f"{tmp}/fold_summaries.csv"))

    def test_fold_column_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            fold_dirs = []
            for i in range(3):
                d = os.path.join(tmp, f"fold{i}")
                os.makedirs(d)
                pd.DataFrame({"metric": [0.1]}).to_csv(f"{d}/summary_results.csv")
                fold_dirs.append(d)

            results = {"save_dir": tmp, "results": fold_dirs}
            ru.collate_folds(results)
            df = pd.read_csv(f"{tmp}/fold_summaries.csv")
            self.assertIn("fold", df.columns)
            self.assertEqual(sorted(df["fold"].unique().tolist()), [0, 1, 2])


class TestSaveSupervisedResult(unittest.TestCase):

    def _run(self, task, n_rows=10, n_labels=2, preds_as_tuple=False, nan_in_ys=False):
        with tempfile.TemporaryDirectory() as tmp:
            split_df = _make_mock_df(n_rows, n_labels)
            if nan_in_ys:
                split_df.ys[0, 0] = np.nan

            preds = np.random.rand(n_rows, n_labels)
            results = {
                "save_dir": tmp,
                "train_preds": (preds, preds) if preds_as_tuple else preds,
                "train_df": split_df,
            }

            metric = MagicMock(return_value=0.42)
            metrics = {"rmse": metric}

            out = ru.save_supervised_result(results, "train", tmp, metrics, task)
            csv_path = f"{tmp}/train_results.csv"
            self.assertTrue(os.path.exists(csv_path))
            return out, pd.read_csv(csv_path, index_col=0)

    def test_group_task_returns_metric_key(self):
        out, _ = self._run("group")
        self.assertIn("rmse", out)

    def test_individual_task_returns_per_label_keys(self):
        out, _ = self._run("individual")
        self.assertIn("label_0_rmse", out)
        self.assertIn("label_1_rmse", out)

    def test_csv_has_pred_columns(self):
        _, df = self._run("group")
        self.assertIn("label_0_pred", df.columns)

    def test_tuple_predictions_uses_pred_idx_zero(self):
        out, _ = self._run("group", preds_as_tuple=True)
        self.assertIn("rmse", out)

    def test_nan_in_labels_handled_for_individual_task(self):
        out, _ = self._run("individual", nan_in_ys=True)
        self.assertTrue(isinstance(out, dict))


class TestSaveResults(unittest.TestCase):

    def test_summary_csv_created_with_all_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = _make_results(tmp)
            processor = MagicMock(return_value={"rmse": 0.1})
            metrics = {"rmse": MagicMock(return_value=0.1)}

            summary = ru.save_results(results, processor, metrics, "group")
            self.assertTrue(os.path.exists(f"{tmp}/summary_results.csv"))
            self.assertEqual(len(summary), 3)

    def test_summary_csv_train_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = _make_results(tmp, include_val=False, include_test=False)
            results["val_df"].__len__ = lambda s: 0
            results["test_df"].__len__ = lambda s: 0

            processor = MagicMock(return_value={"rmse": 0.1})
            metrics = {"rmse": MagicMock(return_value=0.1)}

            summary = ru.save_results(results, processor, metrics, "group")
            self.assertEqual(len(summary), 1)

    def test_processor_called_for_each_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = _make_results(tmp)
            processor = MagicMock(return_value={"rmse": 0.5})
            metrics = {}

            ru.save_results(results, processor, metrics, "group")
            self.assertEqual(processor.call_count, 3)


class TestPlotChannels(unittest.TestCase):

    def test_creates_pdf_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = {"loss": [0.5, 0.4, 0.3], "val_loss": [0.6, 0.5, 0.4]}
            ru.plot_channels(history, ["loss", "val_loss"], "test_plot", tmp)
            self.assertTrue(os.path.exists(f"{tmp}/test_plot.pdf"))

    def test_creates_folder_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "new_folder")
            history = {"loss": [1, 2]}
            ru.plot_channels(history, ["loss"], "myplot", folder)
            self.assertTrue(os.path.isdir(folder))

    def test_single_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = {"acc": [0.7, 0.8]}
            ru.plot_channels(history, ["acc"], "acc_plot", tmp, ylabel="Accuracy")
            self.assertTrue(os.path.exists(f"{tmp}/acc_plot.pdf"))


class TestGetLayers(unittest.TestCase):

    def _make_seq_model(self, layers):
        from keras.models import Sequential as Seq
        model = Seq()
        model.layers = layers
        return model

    def test_flat_model_returns_all_layers(self):
        layer_a = MagicMock(spec=[])
        layer_b = MagicMock(spec=[])
        model = self._make_seq_model([layer_a, layer_b])
        result = ru.get_layers(model)
        self.assertEqual(result, [layer_a, layer_b])

    def test_nested_sequential_flattened(self):
        inner_layer = MagicMock(spec=[])

        from keras.models import Sequential as Seq
        nested = Seq()
        nested.layers = [inner_layer]

        outer_layer = MagicMock(spec=[])
        model = self._make_seq_model([nested, outer_layer])

        result = ru.get_layers(model)
        self.assertIn(inner_layer, result)
        self.assertIn(outer_layer, result)
        self.assertEqual(len(result), 2)

    def test_empty_model_returns_empty_list(self):
        model = self._make_seq_model([])
        result = ru.get_layers(model)
        self.assertEqual(result, [])


class TestGetCoefImportance(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock()
        self.X = np.random.rand(10, 5)
        self.y = np.random.rand(10, 1)
        self.target = 0

    def _patch_mcw(self, attr, return_value=None):
        if return_value is None:
            return_value = np.ones(5)
        return patch.object(ru.mcw, attr, return_value=return_value)

    def test_loss_gradient_calls_get_gradient_weights(self):
        with self._patch_mcw("get_gradient_weights", np.ones(5)) as mock_fn:
            ru.get_coef_importance(self.model, self.X, self.y, self.target, "loss_gradient")
            mock_fn.assert_called_once()

    def test_loss_gradient_signed_calls_get_gradient_weights_signed(self):
        with self._patch_mcw("get_gradient_weights", np.ones(5)) as mock_fn:
            ru.get_coef_importance(self.model, self.X, self.y, self.target, "loss_gradient_signed")
            _, kwargs = mock_fn.call_args
            self.assertTrue(kwargs.get("signed", False))

    def test_gradient_outcome_calls_correct_fn(self):
        with self._patch_mcw("get_weights_gradient_outcome", np.ones(5)) as mock_fn:
            ru.get_coef_importance(self.model, self.X, self.y, self.target, "gradient_outcome")
            mock_fn.assert_called_once()

    def test_permutation_calls_get_permutation_weights(self):
        with self._patch_mcw("get_permutation_weights", np.ones(5)) as mock_fn:
            ru.get_coef_importance(self.model, self.X, self.y, self.target, "permutation")
            mock_fn.assert_called_once()

    def test_linear_calls_get_weights_linear_model(self):
        with self._patch_mcw("get_weights_linear_model", np.ones(5)) as mock_fn:
            ru.get_coef_importance(self.model, self.X, self.y, self.target, "linear")
            mock_fn.assert_called_once()

    def test_unknown_feature_importance_returns_none(self):
        result = ru.get_coef_importance(self.model, self.X, self.y, self.target, "nonexistent_method")
        self.assertIsNone(result)

    def test_shap_method_calls_get_shap_scores(self):
        with self._patch_mcw("get_shap_scores", np.ones(5)) as mock_fn:
            ru.get_coef_importance(self.model, self.X, self.y, self.target, "shap_deep")
            mock_fn.assert_called_once()

    def test_deepexplain_method_calls_get_deep_explain_scores(self):
        with self._patch_mcw("get_deep_explain_scores", np.ones(5)) as mock_fn:
            ru.get_coef_importance(self.model, self.X, self.y, self.target, "deepexplain_deeplift")
            mock_fn.assert_called_once()

    def test_skf_prefix_calls_get_skf_weights(self):
        with self._patch_mcw("get_skf_weights", np.ones(5)) as mock_fn:
            ru.get_coef_importance(self.model, self.X, self.y, self.target, "skf_something")
            mock_fn.assert_called_once()


class TestCollateAggregateResults(unittest.TestCase):

    def _write_results_csv(self, tmp, rows):
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(tmp, "results.csv"))

    def test_creates_aggregated_results_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_results_csv(tmp, [
                {"index": "train", "test_fold": "fold_0", "metric": "rmse", "hyperparams": "h1", "rmse": 0.1},
                {"index": "val",   "test_fold": "fold_0", "metric": "rmse", "hyperparams": "h1", "rmse": 0.2},
            ])
            ru.collate_aggregate_results({"save_dir": tmp})
            self.assertTrue(os.path.exists(os.path.join(tmp, "aggregated_results.csv")))

    def test_output_contains_mean_and_std_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_results_csv(tmp, [
                {"index": "train", "test_fold": "fold_0", "metric": "rmse", "hyperparams": "h1", "rmse": 0.1},
                {"index": "train", "test_fold": "fold_1", "metric": "rmse", "hyperparams": "h1", "rmse": 0.3},
            ])
            ru.collate_aggregate_results({"save_dir": tmp})
            df = pd.read_csv(os.path.join(tmp, "aggregated_results.csv"))
            cols = df.columns.tolist()
            self.assertTrue(any("mean" in c for c in cols), f"Expected a mean column, got {cols}")
            self.assertTrue(any("std" in c for c in cols), f"Expected a std column, got {cols}")
            self.assertTrue(any("sem" in c for c in cols), f"Expected a sem column, got {cols}")

    def test_groups_by_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_results_csv(tmp, [
                {"index": "train", "test_fold": "fold_0", "metric": "rmse", "hyperparams": "h1", "rmse": 0.1},
                {"index": "train", "test_fold": "fold_1", "metric": "rmse", "hyperparams": "h1", "rmse": 0.3},
                {"index": "val",   "test_fold": "fold_0", "metric": "rmse", "hyperparams": "h1", "rmse": 0.5},
                {"index": "val",   "test_fold": "fold_1", "metric": "rmse", "hyperparams": "h1", "rmse": 0.7},
            ])
            ru.collate_aggregate_results({"save_dir": tmp})
            df = pd.read_csv(os.path.join(tmp, "aggregated_results.csv"))
            self.assertEqual(sorted(df["index"].tolist()), ["train", "val"])

    def test_mean_values_are_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_results_csv(tmp, [
                {"index": "train", "test_fold": "fold_0", "metric": "rmse", "hyperparams": "h1", "rmse": 0.2},
                {"index": "train", "test_fold": "fold_1", "metric": "rmse", "hyperparams": "h1", "rmse": 0.4},
            ])
            ru.collate_aggregate_results({"save_dir": tmp})
            df = pd.read_csv(os.path.join(tmp, "aggregated_results.csv"))
            train_row = df[(df["index"] == "train") & (df["performance_metric"] == "rmse")]
            self.assertAlmostEqual(float(train_row["mean"].values[0]), 0.3, places=5)

    def test_non_metric_columns_excluded_from_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_results_csv(tmp, [
                {"index": "train", "test_fold": "fold_0", "metric": "rmse", "hyperparams": "h1", "rmse": 0.1},
                {"index": "train", "test_fold": "fold_1", "metric": "rmse", "hyperparams": "h2", "rmse": 0.3},
            ])
            ru.collate_aggregate_results({"save_dir": tmp})
            df = pd.read_csv(os.path.join(tmp, "aggregated_results.csv"))
            cols = df.columns.tolist()
            for non_metric in ["test_fold", "hyperparams"]:
                self.assertNotIn(non_metric, cols,
                                 f"Non-metric column '{non_metric}' should not appear in aggregated output")
            self.assertEqual(sorted(cols), sorted(["index", "metric", "performance_metric", "mean", "std", "sem"]))

    def test_sem_correct_with_multiple_val_metrics(self):
        # Each fold's results appear once per val_metric (f1/auprc/auc).
        # Without deduplication, n would be n_folds * n_val_metrics, making sem too small.
        # drop_duplicates ensures n = n_folds.
        import math
        fold_values = [0.2, 0.4, 0.6, 0.8, 1.0]
        expected_sem = np.std(fold_values, ddof=1) / math.sqrt(len(fold_values))
        rows = [
            {"index": "test", "test_fold": f"fold_{i}", "metric": m,
             "hyperparams": "h1", "rmse": fold_values[i]}
            for i in range(5)
            for m in ["f1", "auprc", "auc"]
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_results_csv(tmp, rows)
            ru.collate_aggregate_results({"save_dir": tmp})
            df = pd.read_csv(os.path.join(tmp, "aggregated_results.csv"))
            for val_metric in ["f1", "auprc", "auc"]:
                row = df[(df["index"] == "test") & (df["metric"] == val_metric) & (df["performance_metric"] == "rmse")]
                self.assertAlmostEqual(float(row["sem"].values[0]), expected_sem, places=10,
                                       msg=f"sem wrong for val_metric={val_metric}")

    def test_multiple_metric_columns_all_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_results_csv(tmp, [
                {"index": "train", "test_fold": "fold_0", "metric": "rmse", "hyperparams": "h1",
                 "rmse": 0.1, "mae": 0.05},
                {"index": "train", "test_fold": "fold_1", "metric": "rmse", "hyperparams": "h1",
                 "rmse": 0.3, "mae": 0.15},
            ])
            ru.collate_aggregate_results({"save_dir": tmp})
            df = pd.read_csv(os.path.join(tmp, "aggregated_results.csv"))
            self.assertIn("rmse", df["performance_metric"].values)
            self.assertIn("mae", df["performance_metric"].values)

    def test_three_splits_all_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_results_csv(tmp, [
                {"index": split, "test_fold": f"fold_{i}", "metric": "rmse",
                 "hyperparams": "h1", "rmse": 0.1}
                for split in ["train", "val", "test"]
                for i in range(2)
            ])
            ru.collate_aggregate_results({"save_dir": tmp})
            df = pd.read_csv(os.path.join(tmp, "aggregated_results.csv"))
            self.assertEqual(sorted(df["index"].tolist()), ["test", "train", "val"])


if __name__ == "__main__":
    unittest.main(verbosity=2)