import os
import sys
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import logging

# architecture/ and the project root must both be on sys.path:
#   - architecture/  → lets tests do `from pipeline import ...`
#   - project root   → satisfies `from architecture.pnet_model import TFModel` inside pipeline.py
_ARCH_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_ROOT_DIR = os.path.normpath(os.path.join(_ARCH_DIR, ".."))
for _p in (_ARCH_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.disable(logging.CRITICAL)


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_pipeline(config_overrides=None):
    from pipeline import Pipeline
    base = {
        "run_dir": "/tmp",
        "run_id": "test_run",
        "grid_search": [],
        "val_metric": {},
        "fold_collators": [],
        "grid_search_collators": [],
        "feature_selector": MagicMock(),
        "data_augmentor": MagicMock(side_effect=lambda x: x),
        "feature_preprocessor": MagicMock(),
        "results_processors": [],
        "rng_seed": 42,
    }
    if config_overrides:
        base.update(config_overrides)
    p = Pipeline.__new__(Pipeline)
    p.config = base
    p.log = MagicMock()
    p.fold_logger = MagicMock()
    return p


def make_ml_pipeline(config_overrides=None):
    from pipeline import MLPipeline
    base = {
        "run_dir": "/tmp",
        "run_id": "test_run",
        "grid_search": [],
        "val_metric": {},
        "fold_collators": [],
        "grid_search_collators": [],
        "feature_selector": MagicMock(),
        "data_augmentor": MagicMock(side_effect=lambda x: x),
        "feature_preprocessor": MagicMock(),
        "results_processors": [],
        "rng_seed": 42,
        "model": MagicMock(return_value=MagicMock(fit=MagicMock())),
        "task": "regression",
        "model_params": {},
    }
    if config_overrides:
        base.update(config_overrides)
    p = MLPipeline.__new__(MLPipeline)
    p.config = base
    p.log = MagicMock()
    p.fold_logger = MagicMock()
    return p


def make_fold(n=10, n_features=5):
    fold = MagicMock()
    fold.__len__ = MagicMock(return_value=n)
    fold.xs = np.random.rand(n, n_features).astype(np.float32)
    fold.ys = np.random.randint(0, 2, (n, 1)).astype(np.float32)
    fold.get_features = MagicMock(return_value=list(range(n_features)))
    return fold


def make_ext_pipeline(tmp_dir, ext_data=None, model=None, metrics=None,
                      final_model_params=None):
    """
    Build a Pipeline stub with all external-validation dependencies wired up.

    Uses IdentityProcessor for feature_selector and feature_preprocessor so that
    real numpy arrays flow through unchanged, making it easy to assert on output
    files without fighting with MagicMock return values.
    """
    from pipeline import Pipeline, IdentityProcessor

    n_features = 4
    n_ext_samples = 6
    feature_names = [f"feat_{i}" for i in range(n_features)]
    ext_sample_ids = [f"sample_{i}" for i in range(n_ext_samples)]

    full_train = MagicMock()
    full_train.features = feature_names[:]
    full_train.alignment_ids = feature_names[:]
    full_train.__len__ = MagicMock(return_value=10)

    if ext_data is None:
        ext_data = MagicMock()
        ext_data.features = feature_names[:]
        ext_data.alignment_ids = feature_names[:]
        ext_data.ids = ext_sample_ids
        ext_data.xs = np.random.rand(n_ext_samples, n_features).astype(np.float32)
        ext_data.ys = np.random.rand(n_ext_samples, 1).astype(np.float32)
        ext_data.get_labels.return_value = ["label_0"]

    if model is None:
        model = MagicMock()
        model.predict.return_value = np.zeros((n_ext_samples, 1), dtype=np.float32)

    mock_dataloader = MagicMock(return_value=ext_data)

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.run_dir = tmp_dir
    pipeline.log = MagicMock()
    pipeline.data = MagicMock()
    pipeline.data.get_train_test_split.return_value = (full_train, MagicMock())
    pipeline._get_modal_hyperparams = MagicMock(return_value={"C": 1.0})
    pipeline._build_and_train_final_model = MagicMock(return_value=model)

    config = {
        "run_dir": tmp_dir,
        "run_id": "test_run",
        "grid_search": [],
        "feature_selector": IdentityProcessor(),
        "data_augmentor": lambda x: x,
        "feature_preprocessor": IdentityProcessor(),
        "rng_seed": 42,
        "stratified": False,
        "tt_split_seed": 42,
        "shuffle_seed": 42,
        "view_alignment_method": "inner",
        "drop_labels": True,
        "dataloader": mock_dataloader,
        "data_dir": tmp_dir,
        "external_validation_metrics": metrics or {},
        "external_datasets": [
            {
                "tag": "nci60",
                "views": [
                    ("gexpr", "ext.csv", feature_names, 0, lambda x: x, lambda x: x),
                ],
                "labels": [("labels.csv", 0)],
            }
        ],
    }
    if final_model_params is not None:
        config["final_model_params"] = final_model_params

    pipeline.config = config
    return pipeline, model, ext_data


# ── construct_gs_params ───────────────────────────────────────────────────────

class TestConstructGsParams(unittest.TestCase):

    def setUp(self):
        from pipeline import construct_gs_params
        self.fn = construct_gs_params

    def test_single_param_returns_one_entry_per_value(self):
        params = {"model_params": {"a": 1, "b": 2}}
        result = self.fn(params)
        self.assertEqual(len(result), 2)

    def test_single_param_keys_present(self):
        params = {"model_params": {"a": 1, "b": 2}}
        result = self.fn(params)
        keys = {r["model_params_choice"] for r in result}
        self.assertEqual(keys, {"a", "b"})

    def test_single_param_values_present(self):
        params = {"model_params": {"a": 1, "b": 2}}
        result = self.fn(params)
        values = {r["model_params"] for r in result}
        self.assertEqual(values, {1, 2})

    def test_two_params_cartesian_product(self):
        params = {
            "model_params": {"x": 1, "y": 2},
            "lr":           {"low": 0.01, "high": 0.1},
        }
        result = self.fn(params)
        self.assertEqual(len(result), 4)

    def test_two_params_all_combinations_present(self):
        params = {
            "model_params": {"x": 1, "y": 2},
            "lr":           {"low": 0.01, "high": 0.1},
        }
        result = self.fn(params)
        mp_choices = {r["model_params_choice"] for r in result}
        lr_choices  = {r["lr_choice"] for r in result}
        self.assertEqual(mp_choices, {"x", "y"})
        self.assertEqual(lr_choices,  {"low", "high"})

    def test_choice_key_named_correctly(self):
        params = {"my_param": {"opt1": 100}}
        result = self.fn(params)
        self.assertIn("my_param_choice", result[0])
        self.assertEqual(result[0]["my_param_choice"], "opt1")

    def test_single_option_single_result(self):
        params = {"model_params": {"only": 99}}
        result = self.fn(params)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["model_params"], 99)

    def test_each_result_is_dict(self):
        params = {"model_params": {"a": 1}}
        result = self.fn(params)
        self.assertIsInstance(result[0], dict)

    def test_three_params_cartesian_product_size(self):
        params = {
            "p1": {"a": 1, "b": 2},
            "p2": {"c": 3, "d": 4},
            "p3": {"e": 5, "f": 6},
        }
        result = self.fn(params)
        self.assertEqual(len(result), 8)

    def test_values_carried_through_not_just_choice_key(self):
        params = {"lr": {"small": 0.001, "large": 0.1}}
        result = self.fn(params)
        by_choice = {r["lr_choice"]: r["lr"] for r in result}
        self.assertAlmostEqual(by_choice["small"], 0.001)
        self.assertAlmostEqual(by_choice["large"], 0.1)

    def test_two_param_each_combo_has_both_values(self):
        params = {
            "model_params": {"x": 10, "y": 20},
            "lr":           {"low": 0.01, "high": 0.1},
        }
        result = self.fn(params)
        for r in result:
            self.assertIn("model_params", r)
            self.assertIn("lr", r)
            self.assertIn("model_params_choice", r)
            self.assertIn("lr_choice", r)

    def test_no_extra_keys_beyond_param_and_choice(self):
        params = {"alpha": {"a": 1}}
        result = self.fn(params)
        self.assertEqual(set(result[0].keys()), {"alpha", "alpha_choice"})

    def test_already_expanded_list_returned_unchanged(self):
        # If the input is already an expanded list (e.g. second call in run_crossvalidation),
        # construct_gs_params must return it verbatim without re-expanding.
        expanded = [
            {"model_params": {"lr": 0.01}, "model_params_choice": "lr_0.01"},
            {"model_params": {"lr": 0.1},  "model_params_choice": "lr_0.1"},
        ]
        result = self.fn(expanded)
        self.assertIs(result, expanded)


# ── IdentityProcessor ─────────────────────────────────────────────────────────

class TestIdentityProcessor(unittest.TestCase):

    def setUp(self):
        from pipeline import IdentityProcessor
        self.cls = IdentityProcessor

    def test_fit_transform_returns_same_object(self):
        proc = self.cls()
        data = MagicMock()
        self.assertIs(proc.fit_transform(data), data)

    def test_transform_returns_same_object(self):
        proc = self.cls()
        data = MagicMock()
        self.assertIs(proc.transform(data), data)

    def test_fit_returns_none(self):
        proc = self.cls()
        self.assertIsNone(proc.fit(MagicMock()))

    def test_fit_does_not_mutate_input(self):
        proc = self.cls()
        data = MagicMock()
        data.xs = np.array([1.0, 2.0, 3.0])
        proc.fit(data)
        np.testing.assert_array_equal(data.xs, [1.0, 2.0, 3.0])


# ── SKModelWrapper ────────────────────────────────────────────────────────────

class TestSKModelWrapper(unittest.TestCase):

    def setUp(self):
        from pipeline import SKModelWrapper
        self.SKModelWrapper = SKModelWrapper

    def _make_wrapper(self, task="binary classification", model_cls=None):
        if model_cls is None:
            model_cls = MagicMock(return_value=MagicMock(
                predict_proba=MagicMock(return_value=np.array([[0.3, 0.7], [0.6, 0.4]])),
                predict=MagicMock(return_value=np.array([1.0, 2.0])),
                fit=MagicMock()
            ))
        return self.SKModelWrapper(model_cls, task, {})

    def test_init_instantiates_model_with_params(self):
        mock_cls = MagicMock()
        self.SKModelWrapper(mock_cls, "binary classification", {"C": 1.0})
        mock_cls.assert_called_once_with(C=1.0)

    def test_fit_calls_ravel_for_single_label(self):
        wrapper = self._make_wrapper()
        xs = np.random.rand(5, 3)
        ys = np.random.randint(0, 2, (5, 1))
        wrapper.fit(xs, ys)
        self.assertEqual(wrapper.model.fit.call_args[0][1].ndim, 1)

    def test_fit_does_not_ravel_multilabel(self):
        wrapper = self._make_wrapper()
        xs = np.random.rand(5, 3)
        ys = np.random.randint(0, 2, (5, 3))
        wrapper.fit(xs, ys)
        self.assertEqual(wrapper.model.fit.call_args[0][1].ndim, 2)

    def test_predict_binary_returns_positive_class_proba(self):
        wrapper = self._make_wrapper(task="binary classification")
        result = wrapper.predict(np.random.rand(2, 3))
        np.testing.assert_array_almost_equal(result, [0.7, 0.4])

    def test_predict_non_binary_uses_predict_not_predict_proba(self):
        wrapper = self._make_wrapper(task="regression")
        result = wrapper.predict(np.random.rand(2, 3))
        np.testing.assert_array_equal(result, [1.0, 2.0])
        wrapper.model.predict_proba.assert_not_called()

    def test_predict_binary_uses_predict_proba(self):
        wrapper = self._make_wrapper(task="binary classification")
        wrapper.predict(np.random.rand(2, 3))
        wrapper.model.predict_proba.assert_called_once()

    def test_predict_returns_numpy_array(self):
        wrapper = self._make_wrapper()
        self.assertIsInstance(wrapper.predict(np.random.rand(2, 3)), np.ndarray)

    def test_fit_boundary_exactly_one_column_ravels(self):
        wrapper = self._make_wrapper()
        wrapper.fit(np.random.rand(4, 3), np.ones((4, 1)))
        self.assertEqual(wrapper.model.fit.call_args[0][1].ndim, 1)

    def test_fit_boundary_exactly_two_columns_does_not_ravel(self):
        wrapper = self._make_wrapper()
        wrapper.fit(np.random.rand(4, 3), np.ones((4, 2)))
        self.assertEqual(wrapper.model.fit.call_args[0][1].ndim, 2)

    def test_predict_proba_column_index_is_one(self):
        proba = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_cls = MagicMock(return_value=MagicMock(
            predict_proba=MagicMock(return_value=proba),
            fit=MagicMock(),
        ))
        wrapper = self.SKModelWrapper(mock_cls, "binary classification", {})
        np.testing.assert_array_almost_equal(
            wrapper.predict(np.random.rand(2, 3)), proba[:, 1]
        )

    def test_predict_regression_result_shape_preserved(self):
        raw = np.array([3.5, 1.2, 0.9])
        mock_cls = MagicMock(return_value=MagicMock(
            predict=MagicMock(return_value=raw),
            fit=MagicMock(),
        ))
        wrapper = self.SKModelWrapper(mock_cls, "regression", {})
        np.testing.assert_array_equal(wrapper.predict(np.random.rand(3, 2)), raw)


# ── Pipeline._sanitise_config ─────────────────────────────────────────────────

class TestSanitiseConfig(unittest.TestCase):

    def setUp(self):
        self.pipeline = make_pipeline()

    def _wrap(self, inp):
        import json
        raw = self.pipeline._sanitise_config(inp)
        return json.loads("{" + raw + "}")

    def test_int_value_serialised(self):
        self.assertEqual(self._wrap({"epochs": 10})["epochs"], 10)

    def test_float_value_serialised(self):
        self.assertAlmostEqual(self._wrap({"lr": 0.001})["lr"], 0.001)

    def test_string_value_serialised(self):
        self.assertEqual(self._wrap({"name": "my_model"})["name"], "my_model")

    def test_nested_dict_serialised(self):
        self.assertEqual(self._wrap({"outer": {"inner": 42}})["outer"]["inner"], 42)

    def test_list_value_serialised(self):
        self.assertEqual(self._wrap({"sizes": [64, 128, 256]})["sizes"], [64, 128, 256])

    def test_tuple_treated_as_list(self):
        self.assertEqual(self._wrap({"sizes": (64, 128)})["sizes"], [64, 128])

    def test_non_serialisable_falls_back_to_string(self):
        import json
        raw = self.pipeline._sanitise_config({"fn": lambda x: x})
        parsed = json.loads("{" + raw + "}")
        self.assertIn("fn", parsed)

    def test_empty_dict_returns_empty_string(self):
        self.assertEqual(self.pipeline._sanitise_config({}), "")

    def test_empty_list_returns_empty_string(self):
        self.assertEqual(self.pipeline._sanitise_config([]), "")

    def test_bool_value_serialises_without_raising(self):
        import json
        raw = self.pipeline._sanitise_config({"flag": True})
        self.assertIn("flag", json.loads("{" + raw + "}"))

    def test_list_containing_dict_serialised(self):
        import json
        raw = self.pipeline._sanitise_config({"items": [{"k": 1}]})
        self.assertEqual(json.loads("{" + raw + "}")["items"][0]["k"], 1)

    def test_list_containing_non_serialisable_falls_back_to_string(self):
        import json
        raw = self.pipeline._sanitise_config({"fns": [lambda x: x]})
        parsed = json.loads("{" + raw + "}")
        self.assertIsInstance(parsed["fns"][0], str)

    def test_nested_list_inside_list_serialised(self):
        import json
        raw = self.pipeline._sanitise_config({"matrix": [[1, 2], [3, 4]]})
        self.assertEqual(json.loads("{" + raw + "}")["matrix"], [[1, 2], [3, 4]])

    def test_string_with_embedded_quotes_produces_valid_json(self):
        import json
        raw = self.pipeline._sanitise_config({"desc": 'say "hello"'})
        self.assertEqual(json.loads("{" + raw + "}")["desc"], 'say "hello"')

    def test_string_with_embedded_newline_produces_valid_json(self):
        import json
        raw = self.pipeline._sanitise_config({"text": "line1\nline2"})
        parsed = json.loads("{" + raw + "}")
        self.assertIn("line1", parsed["text"])
        self.assertIn("line2", parsed["text"])


# ── Pipeline._fold_run ────────────────────────────────────────────────────────

class TestFoldRun(unittest.TestCase):

    def setUp(self):
        self.val_metric_fn = MagicMock(return_value=0.85)
        self.result_processor = MagicMock()
        self.pipeline = make_pipeline({
            "val_metric": {"auc": self.val_metric_fn},
            "results_processors": [self.result_processor],
        })
        train = make_fold()
        val   = make_fold()
        test  = make_fold()
        fs = self.pipeline.config["feature_selector"]
        fs.fit_transform.return_value = train
        fs.transform.side_effect      = lambda x: x
        pp = self.pipeline.config["feature_preprocessor"]
        pp.fit_transform.return_value = train
        pp.transform.side_effect      = lambda x: x
        self.train, self.val, self.test = train, val, test
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(10)
        self.pipeline._train = MagicMock(return_value=(mock_model, {"loss": [0.1]}))

    def test_returns_val_metric_dict(self):
        result = self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        self.assertIsInstance(result, dict)
        self.assertIn("auc", result)

    def test_val_metric_value_correct(self):
        result = self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        self.assertEqual(result["auc"], 0.85)

    def test_result_processor_called(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        self.result_processor.assert_called_once()

    def test_result_processor_receives_expected_keys(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        passed = self.result_processor.call_args[0][0]
        for key in ("train_preds", "val_preds", "train_df", "val_df", "save_dir", "model"):
            self.assertIn(key, passed)

    def test_feature_selector_fit_transform_called_on_train(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        self.pipeline.config["feature_selector"].fit_transform.assert_called_once_with(self.train)

    def test_no_val_metric_returned_when_val_fold_empty(self):
        result = self.pipeline._fold_run("/tmp/fold", self.train, [], [])
        self.assertIsNone(result)

    def test_data_augmentor_called_exactly_once(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        self.pipeline.config["data_augmentor"].assert_called_once()

    def test_data_augmentor_called_with_train_fold(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        self.pipeline.config["data_augmentor"].assert_called_once_with(self.train)

    def test_feature_selector_transform_called_on_nonempty_val(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        self.pipeline.config["feature_selector"].transform.assert_called()

    def test_feature_selector_transform_called_on_nonempty_test(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, self.test)
        self.assertGreaterEqual(
            self.pipeline.config["feature_selector"].transform.call_count, 2
        )

    def test_preprocessor_fit_transform_called_on_train(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        self.pipeline.config["feature_preprocessor"].fit_transform.assert_called_once_with(self.train)

    def test_preprocessor_transform_called_on_nonempty_val(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        self.pipeline.config["feature_preprocessor"].transform.assert_called()

    def test_preprocessor_transform_called_on_nonempty_test(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, self.test)
        self.assertGreaterEqual(
            self.pipeline.config["feature_preprocessor"].transform.call_count, 2
        )

    def test_results_processor_receives_none_test_preds_when_no_test(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        self.assertIsNone(self.result_processor.call_args[0][0]["test_preds"])

    def test_results_processor_receives_none_val_preds_when_no_val(self):
        pipeline = make_pipeline({"val_metric": {}, "results_processors": [self.result_processor]})
        fs = pipeline.config["feature_selector"]
        fs.fit_transform.return_value = self.train
        fs.transform.side_effect = lambda x: x
        pp = pipeline.config["feature_preprocessor"]
        pp.fit_transform.return_value = self.train
        pp.transform.side_effect = lambda x: x
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(10)
        pipeline._train = MagicMock(return_value=(mock_model, {}))
        pipeline._fold_run("/tmp/fold", self.train, [], [])
        self.assertIsNone(self.result_processor.call_args[0][0]["val_preds"])

    def test_multiple_results_processors_all_called(self):
        proc_a, proc_b = MagicMock(), MagicMock()
        pipeline = make_pipeline({"val_metric": {}, "results_processors": [proc_a, proc_b]})
        fs = pipeline.config["feature_selector"]
        fs.fit_transform.return_value = self.train
        fs.transform.side_effect = lambda x: x
        pp = pipeline.config["feature_preprocessor"]
        pp.fit_transform.return_value = self.train
        pp.transform.side_effect = lambda x: x
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(10)
        pipeline._train = MagicMock(return_value=(mock_model, {}))
        pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        proc_a.assert_called_once()
        proc_b.assert_called_once()

    def test_val_metric_fn_receives_results_dict(self):
        captured = []
        def capturing_metric(results):
            captured.append(results)
            return 0.9
        pipeline = make_pipeline({"val_metric": {"auc": capturing_metric}, "results_processors": []})
        fs = pipeline.config["feature_selector"]
        fs.fit_transform.return_value = self.train
        fs.transform.side_effect = lambda x: x
        pp = pipeline.config["feature_preprocessor"]
        pp.fit_transform.return_value = self.train
        pp.transform.side_effect = lambda x: x
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(10)
        pipeline._train = MagicMock(return_value=(mock_model, {}))
        pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        self.assertEqual(len(captured), 1)
        self.assertIn("train_preds", captured[0])
        self.assertIn("val_preds",   captured[0])

    def test_train_called_with_transformed_folds_not_originals(self):
        transformed_train = make_fold()
        transformed_val   = make_fold()
        pipeline = make_pipeline({"val_metric": {}, "results_processors": []})
        fs = pipeline.config["feature_selector"]
        fs.fit_transform.return_value = transformed_train
        fs.transform.side_effect = lambda x: transformed_val if x is self.val else x
        pp = pipeline.config["feature_preprocessor"]
        pp.fit_transform.return_value = transformed_train
        pp.transform.side_effect = lambda x: transformed_val if x is transformed_val else x
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(10)
        pipeline._train = MagicMock(return_value=(mock_model, {}))
        pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        call_train, call_val = pipeline._train.call_args[0]
        self.assertIs(call_train, transformed_train)
        self.assertIs(call_val,   transformed_val)


# ── Pipeline._build_and_train_final_model (base) ──────────────────────────────

class TestPipelineBuildAndTrainFinalModelBase(unittest.TestCase):

    def test_raises_not_implemented_error(self):
        pipeline = make_pipeline()
        with self.assertRaises(NotImplementedError):
            pipeline._build_and_train_final_model(MagicMock(), MagicMock(), {})


# ── MLPipeline._train ─────────────────────────────────────────────────────────

class TestMLPipelineTrain(unittest.TestCase):

    def setUp(self):
        self.mock_model_instance = MagicMock()
        self.mock_model_cls = MagicMock(return_value=self.mock_model_instance)
        self.pipeline = make_ml_pipeline({
            "model": self.mock_model_cls,
            "task": "regression",
            "model_params": {"alpha": 0.5},
        })
        self.train = make_fold()
        self.val   = make_fold()

    def test_returns_tuple_of_model_and_none(self):
        _, hx = self.pipeline._train(self.train, self.val)
        self.assertIsNone(hx)

    def test_model_instantiated_with_model_params(self):
        self.pipeline._train(self.train, self.val)
        self.mock_model_cls.assert_called_once_with(alpha=0.5)

    def test_model_fit_called_once(self):
        self.pipeline._train(self.train, self.val)
        self.mock_model_instance.fit.assert_called_once()

    def test_rng_seed_set_before_training(self):
        with patch("pipeline.np.random.seed") as mock_seed:
            self.pipeline._train(self.train, self.val)
        mock_seed.assert_called_once_with(42)


# ── MLPipeline._build_and_train_final_model ───────────────────────────────────

class TestMLPipelineBuildAndTrainFinalModel(unittest.TestCase):

    def setUp(self):
        from pipeline import SKModelWrapper
        self.SKModelWrapper = SKModelWrapper
        self.mock_model_instance = MagicMock()
        self.mock_model_cls = MagicMock(return_value=self.mock_model_instance)
        self.pipeline = make_ml_pipeline({
            "model": self.mock_model_cls,
            "task": "regression",
        })
        self.train = make_fold()

    def test_returns_sk_model_wrapper(self):
        result = self.pipeline._build_and_train_final_model(
            self.train, MagicMock(), {"alpha": 0.1}
        )
        self.assertIsInstance(result, self.SKModelWrapper)

    def test_model_instantiated_with_given_params_not_config_params(self):
        # model_params arg must override whatever is in config
        self.pipeline._build_and_train_final_model(
            self.train, MagicMock(), {"C": 99.0}
        )
        self.mock_model_cls.assert_called_once_with(C=99.0)

    def test_model_fit_called_once(self):
        self.pipeline._build_and_train_final_model(self.train, MagicMock(), {})
        self.mock_model_instance.fit.assert_called_once()

    def test_rng_seed_set(self):
        with patch("pipeline.np.random.seed") as mock_seed:
            self.pipeline._build_and_train_final_model(self.train, MagicMock(), {})
        mock_seed.assert_called_once_with(42)


# ── TFPipeline._build_and_train_final_model ───────────────────────────────────

class TestTFPipelineBuildAndTrainFinalModel(unittest.TestCase):

    def _make_tf_pipeline(self, rng_seed=42):
        from pipeline import TFPipeline
        p = TFPipeline.__new__(TFPipeline)
        p.config = {
            "run_id": "tf_test",
            "model": MagicMock(),
            "fitting_params": {"epochs": 5},
            "rng_seed": rng_seed,
        }
        p.log = MagicMock()
        return p

    def test_creates_tf_model_with_correct_arguments(self):
        pipeline = self._make_tf_pipeline()
        mock_tf_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.fit.return_value = (MagicMock(), {})
        mock_tf_cls.return_value = mock_instance

        full_train = MagicMock()
        empty_val  = MagicMock()
        model_params = {"units": 64}

        with patch("pipeline.TFModel", mock_tf_cls):
            pipeline._build_and_train_final_model(full_train, empty_val, model_params)

        mock_tf_cls.assert_called_once_with(
            "tf_test",
            pipeline.config["model"],
            {"units": 64},
            {"epochs": 5},
        )

    def test_fit_called_with_full_train_empty_val_and_seed(self):
        pipeline = self._make_tf_pipeline(rng_seed=7)
        mock_tf_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.fit.return_value = (MagicMock(), {})
        mock_tf_cls.return_value = mock_instance

        full_train = MagicMock()
        empty_val  = MagicMock()

        with patch("pipeline.TFModel", mock_tf_cls):
            pipeline._build_and_train_final_model(full_train, empty_val, {})

        mock_instance.fit.assert_called_once_with(full_train, empty_val, 7)

    def test_returns_model_not_training_history(self):
        pipeline = self._make_tf_pipeline()
        expected_model = MagicMock()
        mock_tf_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.fit.return_value = (expected_model, {"loss": [0.5]})
        mock_tf_cls.return_value = mock_instance

        with patch("pipeline.TFModel", mock_tf_cls):
            result = pipeline._build_and_train_final_model(
                MagicMock(), MagicMock(), {}
            )

        self.assertIs(result, expected_model)

    def test_does_not_reuse_nn_model_instance_from_fold_training(self):
        # Each call to _build_and_train_final_model must create a fresh TFModel,
        # never reuse self.nn_model from fold training.
        pipeline = self._make_tf_pipeline()
        pipeline.nn_model = MagicMock()  # simulate leftover state from fold training

        mock_tf_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.fit.return_value = (MagicMock(), {})
        mock_tf_cls.return_value = mock_instance

        with patch("pipeline.TFModel", mock_tf_cls):
            pipeline._build_and_train_final_model(MagicMock(), MagicMock(), {})

        # TFModel constructor must have been called (not skipped in favour of nn_model)
        mock_tf_cls.assert_called_once()
        # The old nn_model.fit must not have been called
        pipeline.nn_model.fit.assert_not_called()


# ── Pipeline._get_modal_hyperparams ──────────────────────────────────────────

class TestGetModalHyperparams(unittest.TestCase):

    _GRID = [
        {"model_params_choice": "a", "model_params": {"lr": 0.01}},
        {"model_params_choice": "b", "model_params": {"lr": 0.001}},
        {"model_params_choice": "c", "model_params": {"lr": 0.0001}},
    ]

    def _pipeline_with_results(self, rows, grid=None):
        tmp = tempfile.mkdtemp()
        pd.DataFrame(rows).to_csv(os.path.join(tmp, "results.csv"))
        p = make_pipeline({"grid_search": grid or self._GRID})
        p.run_dir = tmp
        return p, tmp

    def test_clear_modal_choice_returned(self):
        rows = [
            {"test_fold": "test_0", "hyperparams": "a"},
            {"test_fold": "test_1", "hyperparams": "a"},
            {"test_fold": "test_2", "hyperparams": "b"},
        ]
        p, tmp = self._pipeline_with_results(rows)
        try:
            self.assertEqual(p._get_modal_hyperparams(), {"lr": 0.01})
        finally:
            shutil.rmtree(tmp)

    def test_tie_breaks_to_fold_0_choice(self):
        # Fold 0 chose "a", fold 1 chose "b" — equal frequency, so fold 0 wins.
        rows = [
            {"test_fold": "test_0", "hyperparams": "a"},
            {"test_fold": "test_1", "hyperparams": "b"},
        ]
        p, tmp = self._pipeline_with_results(rows)
        try:
            self.assertEqual(p._get_modal_hyperparams(), {"lr": 0.01})
        finally:
            shutil.rmtree(tmp)

    def test_tie_not_fold_0_when_fold_1_has_mode(self):
        # Three folds: fold 0 chose "a", folds 1 and 2 chose "b" → "b" wins outright.
        rows = [
            {"test_fold": "test_0", "hyperparams": "a"},
            {"test_fold": "test_1", "hyperparams": "b"},
            {"test_fold": "test_2", "hyperparams": "b"},
        ]
        p, tmp = self._pipeline_with_results(rows)
        try:
            self.assertEqual(p._get_modal_hyperparams(), {"lr": 0.001})
        finally:
            shutil.rmtree(tmp)

    def test_returns_model_params_dict_not_choice_key_string(self):
        rows = [{"test_fold": "test_0", "hyperparams": "c"}]
        p, tmp = self._pipeline_with_results(rows)
        try:
            result = p._get_modal_hyperparams()
            self.assertIsInstance(result, dict)
            self.assertIn("lr", result)
        finally:
            shutil.rmtree(tmp)

    def test_duplicate_rows_per_fold_deduped_before_mode(self):
        # results.csv has many rows per fold (one per metric/split); dedup must
        # happen before counting so each fold counts only once.
        rows = [
            {"test_fold": "test_0", "hyperparams": "a"},
            {"test_fold": "test_0", "hyperparams": "a"},  # same fold, same choice
            {"test_fold": "test_1", "hyperparams": "b"},
            {"test_fold": "test_1", "hyperparams": "b"},
            {"test_fold": "test_2", "hyperparams": "b"},
            {"test_fold": "test_2", "hyperparams": "b"},
        ]
        p, tmp = self._pipeline_with_results(rows)
        try:
            # "b" wins 2-1 after dedup
            self.assertEqual(p._get_modal_hyperparams(), {"lr": 0.001})
        finally:
            shutil.rmtree(tmp)


# ── Pipeline._run_external_validation ────────────────────────────────────────

class TestRunExternalValidation(unittest.TestCase):

    def test_creates_external_validation_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, _ = make_ext_pipeline(tmp)
            pipeline._run_external_validation()
            self.assertTrue(os.path.isdir(os.path.join(tmp, "external_validation")))

    def test_creates_tag_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, _ = make_ext_pipeline(tmp)
            pipeline._run_external_validation()
            self.assertTrue(
                os.path.isdir(os.path.join(tmp, "external_validation", "nci60"))
            )

    def test_writes_predictions_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, _ = make_ext_pipeline(tmp)
            pipeline._run_external_validation()
            self.assertTrue(
                os.path.exists(
                    os.path.join(tmp, "external_validation", "nci60", "predictions.csv")
                )
            )

    def test_predictions_csv_has_correct_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, ext_data = make_ext_pipeline(tmp)
            pipeline._run_external_validation()
            df = pd.read_csv(
                os.path.join(tmp, "external_validation", "nci60", "predictions.csv"),
                index_col=0,
            )
            self.assertEqual(len(df), len(ext_data.ids))

    def test_writes_metrics_csv_when_metrics_configured(self):
        metrics = {"rmse": MagicMock(return_value=0.5)}
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, _ = make_ext_pipeline(tmp, metrics=metrics)
            pipeline._run_external_validation()
            self.assertTrue(
                os.path.exists(
                    os.path.join(tmp, "external_validation", "nci60", "metrics.csv")
                )
            )

    def test_does_not_write_metrics_csv_when_no_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, _ = make_ext_pipeline(tmp, metrics={})
            pipeline._run_external_validation()
            self.assertFalse(
                os.path.exists(
                    os.path.join(tmp, "external_validation", "nci60", "metrics.csv")
                )
            )

    def test_final_model_params_in_config_bypasses_modal_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            override_params = {"C": 42.0}
            pipeline, _, _ = make_ext_pipeline(tmp, final_model_params=override_params)
            pipeline._run_external_validation()
            pipeline._get_modal_hyperparams.assert_not_called()
            _, _, call_params = pipeline._build_and_train_final_model.call_args[0]
            self.assertEqual(call_params, override_params)

    def test_modal_hyperparams_used_when_no_final_model_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, _ = make_ext_pipeline(tmp)
            pipeline._run_external_validation()
            pipeline._get_modal_hyperparams.assert_called_once()

    def test_missing_ext_features_zero_filled(self):
        # External data that only has a subset of training features; missing
        # columns must be zero-filled (not dropped or left as NaN).
        n_features = 4
        feature_names = [f"feat_{i}" for i in range(n_features)]

        # ext_data has only feat_0 and feat_2; feat_1 and feat_3 are absent.
        ext_data = MagicMock()
        ext_data.features = ["feat_0", "feat_2"]
        ext_data.alignment_ids = ["feat_0", "feat_2"]
        ext_data.ids = [f"s{i}" for i in range(4)]
        ext_data.xs = np.ones((4, 2), dtype=np.float32)
        ext_data.ys = np.zeros((4, 1), dtype=np.float32)
        ext_data.get_labels.return_value = ["label_0"]

        captured = {}
        mock_model = MagicMock()
        def capture_predict(xs):
            captured["xs"] = xs.copy()
            return np.zeros((len(xs), 1), dtype=np.float32)
        mock_model.predict.side_effect = capture_predict

        with tempfile.TemporaryDirectory() as tmp:
            pipeline, _, _ = make_ext_pipeline(
                tmp, ext_data=ext_data, model=mock_model
            )
            pipeline._run_external_validation()

        self.assertIn("xs", captured)
        self.assertEqual(captured["xs"].shape[1], n_features)
        # feat_1 (col 1) and feat_3 (col 3) were absent → must be zero
        np.testing.assert_array_equal(captured["xs"][:, 1], 0.0)
        np.testing.assert_array_equal(captured["xs"][:, 3], 0.0)

    def test_feature_selector_and_preprocessor_transform_not_fit_transform_on_ext_data(self):
        # External data must only have .transform() applied, never .fit_transform(),
        # to avoid data leakage from the external set into the fitted pipeline.
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, mock_model, ext_data = make_ext_pipeline(tmp)

            # Replace processors with call-tracking mocks; patch deepcopy to return
            # the same objects so we observe calls on the original mocks.
            mock_fs = MagicMock()
            mock_fs.fit_transform.side_effect = lambda x: x
            mock_fs.transform.side_effect = lambda x: x
            mock_pp = MagicMock()
            mock_pp.fit_transform.side_effect = lambda x: x
            mock_pp.transform.side_effect = lambda x: x
            pipeline.config["feature_selector"] = mock_fs
            pipeline.config["feature_preprocessor"] = mock_pp

            with patch("pipeline.copy.deepcopy", side_effect=lambda x: x):
                pipeline._run_external_validation()

            # Each processor should be fit_transform'd once (train) and transform'd once (ext).
            self.assertEqual(mock_fs.fit_transform.call_count, 1)
            self.assertEqual(mock_fs.transform.call_count, 1)
            self.assertEqual(mock_pp.fit_transform.call_count, 1)
            self.assertEqual(mock_pp.transform.call_count, 1)


if __name__ == "__main__":
    unittest.main()
