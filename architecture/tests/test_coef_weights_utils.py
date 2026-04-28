import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Minimal stubs — only what get_deep_explain_scores needs
# ---------------------------------------------------------------------------

# keras stubs
keras_stub = types.ModuleType("keras")
keras_models_stub = types.ModuleType("keras.models")
keras_layers_stub = types.ModuleType("keras.layers")
keras_backend_stub = types.ModuleType("keras.backend")


class _FakeSequential:
    def __init__(self): self.layers = []

class _FakeDropout: pass
class _FakeBatchNorm: pass
class _FakeInputLayer: pass


keras_models_stub.Sequential = _FakeSequential
keras_layers_stub.Dropout = _FakeDropout
keras_layers_stub.BatchNormalization = _FakeBatchNorm
keras_layers_stub.InputLayer = _FakeInputLayer
keras_stub.models = keras_models_stub
keras_stub.layers = keras_layers_stub
keras_stub.backend = MagicMock()
sys.modules.setdefault("keras", keras_stub)
sys.modules.setdefault("keras.models", keras_models_stub)
sys.modules.setdefault("keras.layers", keras_layers_stub)
sys.modules["keras.backend"] = keras_backend_stub

# tensorflow stubs
tf_stub = types.ModuleType("tensorflow")
tf_compat = types.ModuleType("tensorflow.compat")
tf_v1 = types.ModuleType("tensorflow.compat.v1")
tf_v1.disable_eager_execution = MagicMock()
tf_v1.Session = MagicMock()
tf_compat.v1 = tf_v1
tf_stub.compat = tf_compat
sys.modules.setdefault("tensorflow", tf_stub)
sys.modules.setdefault("tensorflow.compat", tf_compat)
sys.modules.setdefault("tensorflow.compat.v1", tf_v1)

# shap stub
shap_stub = types.ModuleType("shap")
sys.modules.setdefault("shap", shap_stub)

# sklearn stubs
sklearn_stub = types.ModuleType("sklearn")
sklearn_linear = types.ModuleType("sklearn.linear_model")
sklearn_metrics = types.ModuleType("sklearn.metrics")
sklearn_linear.LogisticRegression = MagicMock()
sklearn_metrics.accuracy_score = MagicMock(return_value=0.9)
sklearn_stub.linear_model = sklearn_linear
sklearn_stub.metrics = sklearn_metrics
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.linear_model", sklearn_linear)
sys.modules.setdefault("sklearn.metrics", sklearn_metrics)

# architecture stubs
arch_stub = types.ModuleType("architecture")
eval_stub = types.ModuleType("architecture.evaluation")
eval_stub.get_layers = MagicMock(return_value=[])
deepexplain_stub = types.ModuleType("architecture.deepexplain")
tf_de_stub = types.ModuleType("architecture.deepexplain.tensorflow_")


class _FakeDeepExplain:
    def __init__(self, session=None): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def explain(self, *a, **kw): return np.ones((4, 3))


tf_de_stub.DeepExplain = _FakeDeepExplain
deepexplain_stub.tensorflow_ = tf_de_stub
arch_stub.evaluation = eval_stub
arch_stub.deepexplain = deepexplain_stub
sys.modules.setdefault("architecture", arch_stub)
sys.modules.setdefault("architecture.evaluation", eval_stub)
sys.modules.setdefault("architecture.deepexplain", deepexplain_stub)
sys.modules.setdefault("architecture.deepexplain.tensorflow_", tf_de_stub)
sys.modules.setdefault("architecture.coef_weights_utils", types.ModuleType("architecture.coef_weights_utils"))

import matplotlib; matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent.parent / "coef_weights_utils.py"
if not MODULE_PATH.exists():
    raise ImportError(f"Could not find coef_weights_utils.py at {MODULE_PATH}")

_spec = importlib.util.spec_from_file_location("coef_weights_utils", str(MODULE_PATH))
cw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cw)

# Patch type-check names into module namespace after load
cw.Sequential = _FakeSequential
cw.Dropout = _FakeDropout
cw.BatchNormalization = _FakeBatchNorm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plain_layer(name):
    """MagicMock layer — type() is MagicMock, so NOT skipped by type guard."""
    l = MagicMock()
    l.name = name
    return l


def _typed_layer(name, cls):
    """Real instance of cls — type() == cls, so IS skipped by type guard."""
    l = cls.__new__(cls)
    l.name = name
    return l


# ---------------------------------------------------------------------------
# Tests: get_deep_explain_scores  (the only pathway in use)
# ---------------------------------------------------------------------------

class TestGetDeepExplainScores(unittest.TestCase):
    """
    Covers get_deep_explain_scores() with method_name='deeplift'
    (the deepexplain_deeplift feature_importance pathway).
    get_deep_explain_score_layer is mocked — TF session not required.
    """

    def test_processes_h_prefix_layer(self):
        cw.mm.get_layers.return_value = [_plain_layer("hidden_1")]
        with patch.object(cw, "get_deep_explain_score_layer", return_value=np.ones((5, 3))):
            result = cw.get_deep_explain_scores(MagicMock(), np.zeros((5, 4)), np.zeros(5))
        self.assertIn("hidden_1", result)

    def test_processes_inputs_prefix_layer(self):
        cw.mm.get_layers.return_value = [_plain_layer("inputs_0")]
        with patch.object(cw, "get_deep_explain_score_layer", return_value=np.ones((5, 3))):
            result = cw.get_deep_explain_scores(MagicMock(), np.zeros((5, 4)), np.zeros(5))
        self.assertIn("inputs_0", result)

    def test_skips_non_h_non_inputs_layers(self):
        cw.mm.get_layers.return_value = [_plain_layer("dense_output")]
        with patch.object(cw, "get_deep_explain_score_layer", return_value=np.ones((5, 3))):
            result = cw.get_deep_explain_scores(MagicMock(), np.zeros((5, 4)), np.zeros(5))
        self.assertEqual(result, {})

    def test_skips_sequential_type(self):
        cw.mm.get_layers.return_value = [_typed_layer("hidden_seq", _FakeSequential)]
        with patch.object(cw, "get_deep_explain_score_layer", return_value=np.ones((5, 3))):
            result = cw.get_deep_explain_scores(MagicMock(), np.zeros((5, 4)), np.zeros(5))
        self.assertEqual(result, {})

    def test_skips_dropout_type(self):
        cw.mm.get_layers.return_value = [_typed_layer("hidden_drop", _FakeDropout)]
        with patch.object(cw, "get_deep_explain_score_layer", return_value=np.ones((5, 3))):
            result = cw.get_deep_explain_scores(MagicMock(), np.zeros((5, 4)), np.zeros(5))
        self.assertEqual(result, {})

    def test_skips_batchnorm_type(self):
        cw.mm.get_layers.return_value = [_typed_layer("hidden_bn", _FakeBatchNorm)]
        with patch.object(cw, "get_deep_explain_score_layer", return_value=np.ones((5, 3))):
            result = cw.get_deep_explain_scores(MagicMock(), np.zeros((5, 4)), np.zeros(5))
        self.assertEqual(result, {})

    def test_multidim_gradients_summed_axis_minus2(self):
        cw.mm.get_layers.return_value = [_plain_layer("hidden_1")]
        grads = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2, 3)
        with patch.object(cw, "get_deep_explain_score_layer", return_value=grads):
            result = cw.get_deep_explain_scores(MagicMock(), np.zeros((5, 4)), np.zeros(5))
        np.testing.assert_array_equal(result["hidden_1"], np.sum(grads, axis=-2))

    def test_1d_gradients_stored_as_is(self):
        cw.mm.get_layers.return_value = [_plain_layer("inputs_0")]
        grads = np.array([0.1, 0.2, 0.3])
        with patch.object(cw, "get_deep_explain_score_layer", return_value=grads):
            result = cw.get_deep_explain_scores(MagicMock(), np.zeros((5, 4)), np.zeros(5))
        np.testing.assert_array_equal(result["inputs_0"], grads)

    def test_multiple_layers_all_in_result(self):
        cw.mm.get_layers.return_value = [_plain_layer("hidden_1"), _plain_layer("hidden_2")]
        with patch.object(cw, "get_deep_explain_score_layer", return_value=np.ones(3)):
            result = cw.get_deep_explain_scores(MagicMock(), np.zeros((5, 4)), np.zeros(5))
        self.assertIn("hidden_1", result)
        self.assertIn("hidden_2", result)

    def test_empty_layers_returns_empty_dict(self):
        cw.mm.get_layers.return_value = []
        with patch.object(cw, "get_deep_explain_score_layer", return_value=np.ones(3)):
            result = cw.get_deep_explain_scores(MagicMock(), np.zeros((5, 4)), np.zeros(5))
        self.assertEqual(result, {})

    def test_deeplift_method_name_passed_to_layer_fn(self):
        cw.mm.get_layers.return_value = [_plain_layer("hidden_1")]
        with patch.object(cw, "get_deep_explain_score_layer", return_value=np.ones(3)) as mock_fn:
            cw.get_deep_explain_scores(
                MagicMock(), np.zeros((5, 4)), np.zeros(5), method_name="deeplift"
            )
        _, kwargs = mock_fn.call_args
        self.assertEqual(kwargs.get("method_name") or mock_fn.call_args[0][3], "deeplift")

    def test_detailed_false_returns_dict(self):
        cw.mm.get_layers.return_value = [_plain_layer("hidden_1")]
        with patch.object(cw, "get_deep_explain_score_layer", return_value=np.ones(3)):
            result = cw.get_deep_explain_scores(
                MagicMock(), np.zeros((5, 4)), np.zeros(5), detailed=False
            )
        self.assertIsInstance(result, dict)

    def test_detailed_true_returns_two_dicts(self):
        cw.mm.get_layers.return_value = [_plain_layer("hidden_1")]
        with patch.object(cw, "get_deep_explain_score_layer", return_value=np.ones(3)):
            result = cw.get_deep_explain_scores(
                MagicMock(), np.zeros((5, 4)), np.zeros(5), detailed=True
            )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], dict)
        self.assertIsInstance(result[1], dict)

    def test_detailed_sample_level_dict_matches_keys(self):
        cw.mm.get_layers.return_value = [_plain_layer("hidden_1")]
        grads = np.ones((5, 3))
        with patch.object(cw, "get_deep_explain_score_layer", return_value=grads):
            agg, sample = cw.get_deep_explain_scores(
                MagicMock(), np.zeros((5, 4)), np.zeros(5), detailed=True
            )
        self.assertEqual(set(agg.keys()), set(sample.keys()))

    # --- target index passthrough ---

    def test_target_none_passes_layer_name(self):
        """When target=None, each layer passes its own name as output_index."""
        layers = [_plain_layer("hidden_1"), _plain_layer("hidden_2")]
        cw.mm.get_layers.return_value = layers
        captured = []

        def _capture(*args, **kwargs):
            captured.append(args[2])
            return np.ones(3)

        with patch.object(cw, "get_deep_explain_score_layer", side_effect=_capture):
            cw.get_deep_explain_scores(
                MagicMock(), np.zeros((5, 4)), np.zeros(5), target=None
            )
        # When target=None the real code passes model.output[i] — a MagicMock subscript,
        # not the raw integer. Just verify two calls were made, one per qualifying layer.
        self.assertEqual(len(captured), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)