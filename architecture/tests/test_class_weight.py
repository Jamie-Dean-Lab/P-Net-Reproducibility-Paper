import unittest

import numpy as np

from prostate_cancer_prediction.configs.pnet_single_split import pnet_single_split_config
from prostate_cancer_prediction.configs.pnet_nested_CV import pnet_nested_CV_config
from prostate_cancer_prediction.configs.pnetfc_nested_CV import pnetfc_nested_CV_config
from prostate_cancer_prediction.configs.pnet_GO_single_split import pnet_GO_single_split_config
from prostate_cancer_prediction.configs.dense_single_layer_single_split import \
    dense_single_layer_single_split_config

# `standardize_weights` is the exact function Keras uses to turn `class_weight`
# into per-sample weights in the graph-mode (v1) training path that P-Net runs
# under (eager execution is disabled at import time by coef_weights_utils).
try:
    from keras.engine import training_utils_v1
except ImportError:  # pragma: no cover - guards against a Keras version bump
    training_utils_v1 = None


# Neural-network configs whose class_weight must reach the loss.
NN_CONFIGS = {
    "pnet_single_split": pnet_single_split_config,
    "pnet_nested_CV": pnet_nested_CV_config,
    "pnetfc_nested_CV": pnetfc_nested_CV_config,
    "pnet_GO_single_split": pnet_GO_single_split_config,
    "dense_single_layer_single_split": dense_single_layer_single_split_config,
}


def _expected_output_heads(config):
    """Number of model output heads = n_hidden_layers + 1."""
    model_params = config["grid_search"]["model_params"]
    first = next(iter(model_params.values()))
    return first["n_hidden_layers"] + 1


class TestClassWeight(unittest.TestCase):
    """Verifies class weighting is configured in a form Keras actually applies.

    Keras `model.fit` only honours `class_weight` entries that are dicts keyed by
    class index; a plain list such as ``[0.75, 1.5]`` is silently dropped, so the
    intended minority-class up-weighting never reaches the loss. These tests guard
    against regressing back to the list-of-lists form.
    """

    def test_configs_use_dict_class_weight(self):
        # Each output head needs a {class_index: weight} dict, not a plain list.
        for name, config in NN_CONFIGS.items():
            with self.subTest(config=name):
                cw = config["fitting_params"]["class_weight"]
                self.assertIsInstance(cw, list,
                                      f"{name}: class_weight should be a per-head list")
                for head in cw:
                    self.assertIsInstance(
                        head, dict,
                        f"{name}: each class_weight entry must be a dict like "
                        f"{{0: 0.75, 1: 1.5}}, got {type(head).__name__}: {head!r}",
                    )
                    self.assertEqual(
                        set(head.keys()), {0, 1},
                        f"{name}: class_weight keys must be class indices {{0, 1}}, "
                        f"got {sorted(head.keys())}",
                    )

    def test_one_class_weight_per_output_head(self):
        # There must be exactly one class_weight entry per output head, all equal.
        for name, config in NN_CONFIGS.items():
            with self.subTest(config=name):
                cw = config["fitting_params"]["class_weight"]
                self.assertEqual(
                    len(cw), _expected_output_heads(config),
                    f"{name}: expected one class_weight per output head",
                )
                self.assertTrue(all(head == cw[0] for head in cw),
                                f"{name}: all heads should share the same class_weight")

    @unittest.skipIf(training_utils_v1 is None,
                     "keras.engine.training_utils_v1 unavailable in this Keras version")
    def test_dict_class_weight_is_applied(self):
        # The dict form is turned into per-sample weights: class 0 -> 0.75, 1 -> 1.5.
        cw = pnet_single_split_config["fitting_params"]["class_weight"][0]
        y = np.array([0, 0, 0, 1, 1], dtype="float32").reshape(-1, 1)

        sample_weight = training_utils_v1.standardize_weights(y, class_weight=cw)

        self.assertIsNotNone(sample_weight,
                             "dict class_weight should produce per-sample weights")
        sample_weight = np.asarray(sample_weight).ravel()
        np.testing.assert_allclose(sample_weight[y.ravel() == 0], cw[0])
        np.testing.assert_allclose(sample_weight[y.ravel() == 1], cw[1])

    @unittest.skipIf(training_utils_v1 is None,
                     "keras.engine.training_utils_v1 unavailable in this Keras version")
    def test_list_class_weight_is_silently_ignored(self):
        # Regression check: the old [0.75, 1.5] list form is dropped (returns None),
        # which is exactly why the dict form is required.
        y = np.array([0, 0, 1, 1], dtype="float32").reshape(-1, 1)

        sample_weight = training_utils_v1.standardize_weights(y, class_weight=[0.75, 1.5])

        self.assertIsNone(sample_weight,
                          "list class_weight is silently ignored by Keras")


if __name__ == "__main__":
    unittest.main()
