import copy
import importlib
import os
import pkgutil
import unittest

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

from prostate_cancer_prediction import configs as configs_package
from prostate_cancer_prediction.configs.dense_single_layer_train_size_variation import \
    dense_single_layer_train_size_variation_configs
from prostate_cancer_prediction.configs.pnet_single_split import pnet_single_split_config
from prostate_cancer_prediction.configs.pnetfc_single_split_elmarakeby import pnetfc_single_split_elmarakeby_config
from architecture.pipeline import TFPipeline
from architecture.pnet_model import compile_pnet
import tensorflow as tf
import numpy as np


# Build models and check they are as described by P-NET's authors in the original paper.

# The assertions below describe the architecture, which is shared by every config in a
# family, so any sparse P-NET / fully connected P-NET config can be substituted here.
# train_model normalises the differences between config shapes (grid-searched vs fixed
# model_params, held-out splits vs cross-validated sample pools).
PNET_CONFIG = pnet_single_split_config
PNETFC_CONFIG = pnetfc_single_split_elmarakeby_config


def collect_pnet_configs():
    """
    Returns every P-NET / P-NET-FC config defined in the configs package as
    (run_id, config) pairs, identified by the model builder they use rather than by
    module name, so renamed or newly added configs are picked up automatically.
    """
    configs = {}
    for module_info in pkgutil.iter_modules(configs_package.__path__):
        module = importlib.import_module(f"{configs_package.__name__}.{module_info.name}")
        for value in vars(module).values():
            # Configs are either defined singly or as a list of variants.
            candidates = value if isinstance(value, list) else [value]
            for candidate in candidates:
                if isinstance(candidate, dict) and candidate.get("model") is compile_pnet:
                    configs.setdefault(candidate["run_id"], candidate)
    return sorted(configs.items())


class TestPnetModels(unittest.TestCase):

    def test_helpers_support_every_pnet_config(self):
        # Guard for the helpers below: the architecture assertions in this file hold for
        # any P-NET config, so no single config should be a hard dependency of the tests.
        pnet_configs = collect_pnet_configs()
        self.assertTrue(pnet_configs, "No P-NET configs found")

        for run_id, config in pnet_configs:
            with self.subTest(config=run_id):
                model_params = self.resolve_model_params(config)
                self.assertIn("sparse", model_params)
                train_ids, _, _ = self.resolve_split_ids(config)
                self.assertTrue(isinstance(train_ids, float) or len(train_ids) > 0,
                                f"Config '{run_id}' selects no training samples")

    def test_pnet_model_attributes(self):
        model, keras_model = self.train_model(PNET_CONFIG)

        # Quick summary
        keras_model.summary()

        # Total parameter count
        total_params = keras_model.count_params()
        print(f"Total parameters: {total_params}")

        # Trainable vs non-trainable (graph execution compatible)
        trainable_params = int(np.sum([tf.keras.backend.count_params(w) for w in keras_model.trainable_weights]))
        non_trainable_params = int(np.sum([tf.keras.backend.count_params(w) for w in keras_model.non_trainable_weights]))
        print(f"Trainable: {trainable_params}")
        print(f"Non-trainable: {non_trainable_params}")

        # Per-layer breakdown
        for layer in keras_model.layers:
            weights = layer.get_weights()
            if weights:
                print(f"Layer '{layer.name}': shapes={[w.shape for w in weights]}, "
                      f"params={sum(w.size for w in weights)}")

        # --- Assertions ---
        self.assertIsNotNone(keras_model)

        # Total trainable parameters
        self.assertEqual(trainable_params, 71009,
                         f"Expected 71009 trainable params, got {trainable_params}")

        # Number of nodes per layer (as described in the P-Net paper)
        EXPECTED_NODES_PER_LAYER = {
            "inputs": 27687,
            "h0":      9229,
            "h1":      1387,
            "h2":      1066,
            "h3":       447,
            "h4":       147,
            "h5":        26,
        }

        layer_map = {layer.name: layer for layer in keras_model.layers}

        for layer_name, expected_nodes in EXPECTED_NODES_PER_LAYER.items():
            self.assertIn(layer_name, layer_map,
                          f"Expected layer '{layer_name}' not found in model")
            output_shape = layer_map[layer_name].output_shape
            # InputLayer wraps its output shape in a list: [(None, 27687)]
            if isinstance(output_shape, list):
                output_shape = output_shape[0]
            actual_nodes = output_shape[-1]
            self.assertEqual(actual_nodes, expected_nodes,
                             f"Layer '{layer_name}': expected {expected_nodes} nodes, got {actual_nodes}")

        # Verify feature names (feature_names is a dict)
        self.assertIn("inputs", model.feature_names)
        self.assertIn("h0", model.feature_names)
        self.assertIn("h1", model.feature_names)

        # Verify loss weights
        self.assertIn("loss_weights", model.model_params)
        self.assertIsNotNone(model.model_params["loss_weights"])
        self.assertEqual(list(model.model_params["loss_weights"]), [2, 7, 20, 54, 148, 400],
                         f"Expected loss_weights [2, 7, 20, 54, 148, 400], "
                         f"got {model.model_params['loss_weights']}")

    def test_fully_connected_pnet_model_attributes(self):
        model, keras_model = self.train_model(PNETFC_CONFIG)

        # Quick summary
        keras_model.summary()

        # Total parameter count
        total_params = keras_model.count_params()
        print(f"Total parameters: {total_params}")

        # Trainable vs non-trainable (graph execution compatible)
        trainable_params = int(np.sum([tf.keras.backend.count_params(w) for w in keras_model.trainable_weights]))
        non_trainable_params = int(
            np.sum([tf.keras.backend.count_params(w) for w in keras_model.non_trainable_weights]))
        print(f"Trainable: {trainable_params}")
        print(f"Non-trainable: {non_trainable_params}")

        # Per-layer breakdown
        for layer in keras_model.layers:
            weights = layer.get_weights()
            if weights:
                print(f"Layer '{layer.name}': shapes={[w.shape for w in weights]}, "
                      f"params={sum(w.size for w in weights)}")

        # --- Assertions ---
        self.assertIsNotNone(keras_model)

        # Total trainable parameters (fully connected has far more than sparse P-Net)
        self.assertEqual(trainable_params, 14877495,
                         f"Expected 14877495 trainable params, got {trainable_params}")

        # Verify it is fully connected — all hidden layers (h0-h5) should be Dense layers
        layer_map = {layer.name: layer for layer in keras_model.layers}
        for layer_name in ["h1", "h2", "h3", "h4", "h5"]:
            self.assertIn(layer_name, layer_map,
                          f"Expected layer '{layer_name}' not found in model")
            self.assertIsInstance(layer_map[layer_name], tf.keras.layers.Dense,
                                  f"Expected '{layer_name}' to be a fully connected Dense layer, "
                                  f"got {type(layer_map[layer_name])}")

        # Number of nodes per layer matches P-Net topology (same structure, different connectivity)
        EXPECTED_NODES_PER_LAYER = {
            "inputs": 27687,
            "h0": 9229,
            "h1": 1387,
            "h2": 1066,
            "h3": 447,
            "h4": 147,
            "h5": 26,
        }

        for layer_name, expected_nodes in EXPECTED_NODES_PER_LAYER.items():
            self.assertIn(layer_name, layer_map,
                          f"Expected layer '{layer_name}' not found in model")
            output_shape = layer_map[layer_name].output_shape
            if isinstance(output_shape, list):
                output_shape = output_shape[0]
            actual_nodes = output_shape[-1]
            self.assertEqual(actual_nodes, expected_nodes,
                             f"Layer '{layer_name}': expected {expected_nodes} nodes, got {actual_nodes}")

        # Verify loss weights
        self.assertIn("loss_weights", model.model_params)
        self.assertIsNotNone(model.model_params["loss_weights"])
        self.assertEqual(list(model.model_params["loss_weights"]), [2, 7, 20, 54, 148, 400],
                         f"Expected loss_weights [2, 7, 20, 54, 148, 400], "
                         f"got {model.model_params['loss_weights']}")


    # N.B we have 83,068 trainable params in the dense single layer, slightly more than in P-NET (71,009)
    def test_pnet_dense_single_layer_model_attributes(self):
        model, keras_model = self.train_model(dense_single_layer_train_size_variation_configs[0])

        # Quick summary
        keras_model.summary()

        # Total parameter count
        total_params = keras_model.count_params()
        print(f"Total parameters: {total_params}")

        # Trainable vs non-trainable (graph execution compatible)
        trainable_params = int(np.sum([tf.keras.backend.count_params(w) for w in keras_model.trainable_weights]))
        non_trainable_params = int(np.sum([tf.keras.backend.count_params(w) for w in keras_model.non_trainable_weights]))
        print(f"Trainable: {trainable_params}")
        print(f"Non-trainable: {non_trainable_params}")

        # Per-layer breakdown
        for layer in keras_model.layers:
            weights = layer.get_weights()
            if weights:
                print(f"Layer '{layer.name}': shapes={[w.shape for w in weights]}, "
                      f"params={sum(w.size for w in weights)}")

        # --- Assertions ---
        self.assertIsNotNone(keras_model)

        # Verify it is a single hidden dense layer network
        hidden_layers = [layer for layer in keras_model.layers
                         if hasattr(layer, 'units') and layer.name != 'output']
        self.assertEqual(len(hidden_layers), 1,
                         f"Expected 1 hidden dense layer, got {len(hidden_layers)}")
        self.assertIsInstance(hidden_layers[0], tf.keras.layers.Dense,
                              f"Expected hidden layer to be Dense, got {type(hidden_layers[0])}")

        # Total trainable parameters
        self.assertEqual(trainable_params, 83068,
                         f"Expected 83068 trainable params, got {trainable_params}")


    def train_model(self, config):
        config = copy.deepcopy(config)
        config["model_params"] = self.resolve_model_params(config)
        train_ids, val_ids, test_ids = self.resolve_split_ids(config)
        pipeline = TFPipeline(config)

        pipeline.log = pipeline._get_logger("main_logger", config["run_dir"])
        pipeline._load_data(shuffle_seed=42)

        train_df, val_df, test_df = pipeline.data.get_specific_split(train_ids,
                                                                     val_ids,
                                                                     test_ids,
                                                                     config["tt_split_seed"])
        train_fold = train_df
        val_fold = val_df
        test_fold = test_df
        feature_selector = config["feature_selector"]

        train_fold = feature_selector.fit_transform(train_fold)
        train_fold = pipeline.config["data_augmentor"](train_fold)
        if len(val_fold) > 0:
            val_fold = feature_selector.transform(val_fold)
        if len(test_fold) > 0:
            test_fold = feature_selector.transform(test_fold)

        preprocessor = pipeline.config["feature_preprocessor"]
        train_fold = preprocessor.fit_transform(train_fold)
        val_fold = preprocessor.transform(val_fold)
        model, _ = pipeline._train(train_fold, val_fold)

        return model, model.predictor

    @staticmethod
    def resolve_model_params(config):
        """
        _train reads config["model_params"], but configs that grid search over
        hyperparameters nest them under config["grid_search"] instead. Any single
        grid entry builds the same architecture, so take the first one.
        """
        if "model_params" in config:
            return config["model_params"]
        grid_search = config.get("grid_search")
        if isinstance(grid_search, dict) and grid_search.get("model_params"):
            first_key = next(iter(grid_search["model_params"]))
            return grid_search["model_params"][first_key]
        raise ValueError(f"Config '{config.get('run_id')}' defines no model_params, either "
                         f"directly or under grid_search")

    @staticmethod
    def resolve_split_ids(config):
        """
        Returns the (train, val, test) sample ids to build the model from. Single split
        configs name all three explicitly, whereas crossvalidation configs generate their
        own folds instead — they either restrict the sample pool with "samples_to_include"
        or name no samples at all, in which case the whole dataset is used for training.
        Only the training split matters here: it just has to be non-empty to fit on.
        """
        train_ids = config.get("samples_to_include", config.get("train_samples", 1.0))
        return train_ids, config.get("val_samples", []), config.get("test_samples", [])




if __name__ == '__main__':
    unittest.main()