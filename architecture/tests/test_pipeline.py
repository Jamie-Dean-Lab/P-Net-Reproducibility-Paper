import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open
import logging

# Suppress logging output during tests
logging.disable(logging.CRITICAL)


# ── Helpers / Stubs ──────────────────────────────────────────────────────────

def make_pipeline(config_overrides=None):
    """Create a minimal Pipeline instance without importing TFModel."""
    from pipeline import Pipeline
    base_config = {
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
        base_config.update(config_overrides)
    p = Pipeline.__new__(Pipeline)
    p.config = base_config
    p.log = MagicMock()
    p.fold_logger = MagicMock()
    return p


def make_fold(n=10, n_features=5):
    """Return a mock dataset fold."""
    fold = MagicMock()
    fold.__len__ = MagicMock(return_value=n)
    fold.xs = np.random.rand(n, n_features)
    fold.ys = np.random.randint(0, 2, (n, 1))
    fold.get_features = MagicMock(return_value=list(range(n_features)))
    return fold


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
        # The actual param value must appear under the param key, not just the
        # choice label under the _choice key.
        params = {"lr": {"small": 0.001, "large": 0.1}}
        result = self.fn(params)
        by_choice = {r["lr_choice"]: r["lr"] for r in result}
        self.assertAlmostEqual(by_choice["small"], 0.001)
        self.assertAlmostEqual(by_choice["large"], 0.1)

    def test_two_param_each_combo_has_both_values(self):
        # Every result dict must contain both param keys so that config
        # updates in the pipeline loop can apply all values at once.
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
        # Only the param key and its corresponding _choice key should be present.
        params = {"alpha": {"a": 1}}
        result = self.fn(params)
        self.assertEqual(set(result[0].keys()), {"alpha", "alpha_choice"})


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
        call_args = wrapper.model.fit.call_args
        # second arg should be 1-D (ravelled)
        self.assertEqual(call_args[0][1].ndim, 1)

    def test_fit_does_not_ravel_multilabel(self):
        wrapper = self._make_wrapper()
        xs = np.random.rand(5, 3)
        ys = np.random.randint(0, 2, (5, 3))
        wrapper.fit(xs, ys)
        call_args = wrapper.model.fit.call_args
        self.assertEqual(call_args[0][1].ndim, 2)

    def test_predict_binary_returns_positive_class_proba(self):
        wrapper = self._make_wrapper(task="binary classification")
        xs = np.random.rand(2, 3)
        result = wrapper.predict(xs)
        np.testing.assert_array_almost_equal(result, [0.7, 0.4])

    def test_predict_non_binary_uses_predict_not_predict_proba(self):
        wrapper = self._make_wrapper(task="regression")
        xs = np.random.rand(2, 3)
        result = wrapper.predict(xs)
        np.testing.assert_array_equal(result, [1.0, 2.0])
        wrapper.model.predict_proba.assert_not_called()

    def test_predict_binary_uses_predict_proba(self):
        wrapper = self._make_wrapper(task="binary classification")
        xs = np.random.rand(2, 3)
        wrapper.predict(xs)
        wrapper.model.predict_proba.assert_called_once()

    def test_predict_returns_numpy_array(self):
        wrapper = self._make_wrapper()
        result = wrapper.predict(np.random.rand(2, 3))
        self.assertIsInstance(result, np.ndarray)

    def test_fit_boundary_exactly_one_column_ravels(self):
        # The branch condition is `ys.shape[1] > 1`, so shape (n, 1) must ravel.
        wrapper = self._make_wrapper()
        xs = np.random.rand(4, 3)
        ys = np.ones((4, 1))
        wrapper.fit(xs, ys)
        self.assertEqual(wrapper.model.fit.call_args[0][1].ndim, 1)

    def test_fit_boundary_exactly_two_columns_does_not_ravel(self):
        # shape (n, 2) is the smallest multilabel case — must NOT ravel.
        wrapper = self._make_wrapper()
        xs = np.random.rand(4, 3)
        ys = np.ones((4, 2))
        wrapper.fit(xs, ys)
        self.assertEqual(wrapper.model.fit.call_args[0][1].ndim, 2)

    def test_predict_proba_column_index_is_one(self):
        # predict_proba returns shape (n, 2); the wrapper must slice [:, 1]
        # (positive class), not [:, 0].
        proba = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_cls = MagicMock(return_value=MagicMock(
            predict_proba=MagicMock(return_value=proba),
            fit=MagicMock(),
        ))
        wrapper = self.SKModelWrapper(mock_cls, "binary classification", {})
        result = wrapper.predict(np.random.rand(2, 3))
        np.testing.assert_array_almost_equal(result, proba[:, 1])

    def test_predict_regression_result_shape_preserved(self):
        # For non-binary tasks, whatever model.predict returns comes back unchanged.
        raw = np.array([3.5, 1.2, 0.9])
        mock_cls = MagicMock(return_value=MagicMock(
            predict=MagicMock(return_value=raw),
            fit=MagicMock(),
        ))
        wrapper = self.SKModelWrapper(mock_cls, "regression", {})
        result = wrapper.predict(np.random.rand(3, 2))
        np.testing.assert_array_equal(result, raw)


# ── Pipeline._sanitise_config ─────────────────────────────────────────────────

class TestSanitiseConfig(unittest.TestCase):

    def setUp(self):
        self.pipeline = make_pipeline()

    def _wrap(self, inp):
        """Wrap output in braces so it's valid JSON, then parse it."""
        import json
        raw = self.pipeline._sanitise_config(inp)
        return json.loads("{" + raw + "}")

    def test_int_value_serialised(self):
        result = self._wrap({"epochs": 10})
        self.assertEqual(result["epochs"], 10)

    def test_float_value_serialised(self):
        result = self._wrap({"lr": 0.001})
        self.assertAlmostEqual(result["lr"], 0.001)

    def test_string_value_serialised(self):
        result = self._wrap({"name": "my_model"})
        self.assertEqual(result["name"], "my_model")

    def test_nested_dict_serialised(self):
        result = self._wrap({"outer": {"inner": 42}})
        self.assertEqual(result["outer"]["inner"], 42)

    def test_list_value_serialised(self):
        result = self._wrap({"sizes": [64, 128, 256]})
        self.assertEqual(result["sizes"], [64, 128, 256])

    def test_tuple_treated_as_list(self):
        result = self._wrap({"sizes": (64, 128)})
        self.assertEqual(result["sizes"], [64, 128])

    def test_non_serialisable_falls_back_to_string(self):
        import json
        # A lambda is not JSON-serialisable; should not raise
        raw = self.pipeline._sanitise_config({"fn": lambda x: x})
        parsed = json.loads("{" + raw + "}")
        self.assertIn("fn", parsed)

    def test_empty_dict_returns_empty_string(self):
        result = self.pipeline._sanitise_config({})
        self.assertEqual(result, "")

    def test_empty_list_returns_empty_string(self):
        result = self.pipeline._sanitise_config([])
        self.assertEqual(result, "")

    def test_bool_value_serialises_as_integer(self):
        # bool is a subclass of int, so the int branch fires and it round-trips
        # as 1 / 0 rather than true / false.  Either is acceptable JSON; what
        # matters is that it doesn't raise.
        import json
        raw = self.pipeline._sanitise_config({"flag": True})
        parsed = json.loads("{" + raw + "}")
        self.assertIn("flag", parsed)

    def test_list_containing_dict_serialised(self):
        # The list-branch has a dict sub-case; verify it produces valid JSON.
        import json
        raw = self.pipeline._sanitise_config({"items": [{"k": 1}]})
        parsed = json.loads("{" + raw + "}")
        self.assertEqual(parsed["items"][0]["k"], 1)

    def test_list_containing_non_serialisable_falls_back_to_string(self):
        # The else-branch inside the list handler should stringify without raising.
        import json
        raw = self.pipeline._sanitise_config({"fns": [lambda x: x]})
        parsed = json.loads("{" + raw + "}")
        self.assertIn("fns", parsed)
        self.assertIsInstance(parsed["fns"][0], str)

    def test_nested_list_inside_list_serialised(self):
        import json
        raw = self.pipeline._sanitise_config({"matrix": [[1, 2], [3, 4]]})
        parsed = json.loads("{" + raw + "}")
        self.assertEqual(parsed["matrix"], [[1, 2], [3, 4]])

    def test_string_with_embedded_quotes_produces_valid_json(self):
        import json
        raw = self.pipeline._sanitise_config({"desc": 'say "hello"'})
        parsed = json.loads("{" + raw + "}")
        self.assertEqual(parsed["desc"], 'say "hello"')

    def test_string_with_embedded_newline_produces_valid_json(self):
        import json
        raw = self.pipeline._sanitise_config({"text": "line1\nline2"})
        parsed = json.loads("{" + raw + "}")
        self.assertIn("text", parsed)

    def test_string_with_embedded_newline_value_preserved(self):
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
        # Mock feature selector and preprocessor to pass folds through unchanged
        train = make_fold()
        val   = make_fold()
        test  = make_fold()
        fs = self.pipeline.config["feature_selector"]
        fs.fit_transform.return_value = train
        fs.transform.side_effect      = lambda x: x  # pass through so [] stays []
        pp = self.pipeline.config["feature_preprocessor"]
        pp.fit_transform.return_value = train
        pp.transform.side_effect      = lambda x: x  # pass through so [] stays []
        self.train, self.val, self.test = train, val, test

        # Mock _train to return a model with a predict method
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
        # Augmentor receives the output of feature_selector.fit_transform
        # (which our stub returns as self.train).
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        self.pipeline.config["data_augmentor"].assert_called_once_with(self.train)

    def test_feature_selector_transform_called_on_nonempty_val(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        fs = self.pipeline.config["feature_selector"]
        # transform must have been called at least once (for val)
        fs.transform.assert_called()

    def test_feature_selector_transform_called_on_nonempty_test(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, self.test)
        fs = self.pipeline.config["feature_selector"]
        # called for both val and test
        self.assertGreaterEqual(fs.transform.call_count, 2)

    def test_preprocessor_fit_transform_called_on_train(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        pp = self.pipeline.config["feature_preprocessor"]
        pp.fit_transform.assert_called_once_with(self.train)

    def test_preprocessor_transform_called_on_nonempty_val(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        pp = self.pipeline.config["feature_preprocessor"]
        pp.transform.assert_called()

    def test_preprocessor_transform_called_on_nonempty_test(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, self.test)
        pp = self.pipeline.config["feature_preprocessor"]
        self.assertGreaterEqual(pp.transform.call_count, 2)

    def test_results_processor_receives_none_test_preds_when_no_test(self):
        self.pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        passed = self.result_processor.call_args[0][0]
        self.assertIsNone(passed["test_preds"])

    def test_results_processor_receives_none_val_preds_when_no_val(self):
        pipeline = make_pipeline({
            "val_metric": {},
            "results_processors": [self.result_processor],
        })
        fs = pipeline.config["feature_selector"]
        fs.fit_transform.return_value = self.train
        fs.transform.side_effect = lambda x: x
        pp = pipeline.config["feature_preprocessor"]
        pp.fit_transform.return_value = self.train
        pp.transform.side_effect = lambda x: x
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(10)
        pipeline._train = MagicMock(return_value=(mock_model, {"loss": [0.1]}))
        pipeline._fold_run("/tmp/fold", self.train, [], [])
        passed = self.result_processor.call_args[0][0]
        self.assertIsNone(passed["val_preds"])

    def test_multiple_results_processors_all_called(self):
        proc_a = MagicMock()
        proc_b = MagicMock()
        pipeline = make_pipeline({
            "val_metric": {},
            "results_processors": [proc_a, proc_b],
        })
        fs = pipeline.config["feature_selector"]
        fs.fit_transform.return_value = self.train
        fs.transform.side_effect = lambda x: x
        pp = pipeline.config["feature_preprocessor"]
        pp.fit_transform.return_value = self.train
        pp.transform.side_effect = lambda x: x
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(10)
        pipeline._train = MagicMock(return_value=(mock_model, {"loss": [0.1]}))
        pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        proc_a.assert_called_once()
        proc_b.assert_called_once()

    def test_val_metric_fn_receives_results_dict(self):
        # The metric function is called as v(results); verify it receives a dict
        # with at least the standard keys rather than positional args.
        captured = []
        def capturing_metric(results):
            captured.append(results)
            return 0.9
        pipeline = make_pipeline({
            "val_metric": {"auc": capturing_metric},
            "results_processors": [],
        })
        fs = pipeline.config["feature_selector"]
        fs.fit_transform.return_value = self.train
        fs.transform.side_effect = lambda x: x
        pp = pipeline.config["feature_preprocessor"]
        pp.fit_transform.return_value = self.train
        pp.transform.side_effect = lambda x: x
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(10)
        pipeline._train = MagicMock(return_value=(mock_model, {"loss": [0.1]}))
        pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        self.assertEqual(len(captured), 1)
        self.assertIn("train_preds", captured[0])
        self.assertIn("val_preds",   captured[0])

    def test_train_called_with_transformed_folds_not_originals(self):
        # If the feature selector or preprocessor modify the fold, _train must
        # receive the transformed version, not the raw input.
        transformed_train = make_fold()
        transformed_val   = make_fold()
        pipeline = make_pipeline({
            "val_metric": {},
            "results_processors": [],
        })
        fs = pipeline.config["feature_selector"]
        fs.fit_transform.return_value = transformed_train
        fs.transform.side_effect = lambda x: transformed_val if x is self.val else x
        pp = pipeline.config["feature_preprocessor"]
        pp.fit_transform.return_value = transformed_train
        pp.transform.side_effect = lambda x: transformed_val if x is transformed_val else x
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(10)
        pipeline._train = MagicMock(return_value=(mock_model, {"loss": [0.1]}))
        pipeline._fold_run("/tmp/fold", self.train, self.val, [])
        call_train, call_val = pipeline._train.call_args[0]
        self.assertIs(call_train, transformed_train)
        self.assertIs(call_val,   transformed_val)


if __name__ == "__main__":
    unittest.main()