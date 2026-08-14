import unittest

import numpy as np
import pandas as pd
from keras import Input, Model

from architecture.layers_custom import Diagonal, SparseTF


def _diagonal(units, n_features, use_bias=True, activation=None,
              kernel_initializer="ones", bias_initializer="zeros"):
    """Wrap a Diagonal layer in a Model so it evaluates in eager or graph mode."""
    inp = Input(shape=(n_features,), dtype="float32")
    layer = Diagonal(units, activation, use_bias, kernel_initializer, bias_initializer,
                     None, None, None, None, name="h0")
    return Model(inp, layer(inp)), layer


def _sparse(map_df, use_bias=False, activation=None, kernel_initializer="ones"):
    inp = Input(shape=(map_df.shape[0],), dtype="float32")
    layer = SparseTF(map_df.shape[1], map_df, None, kernel_initializer, None, activation,
                     use_bias, "zeros", None, None, None, name="h1")
    return Model(inp, layer(inp)), layer


def _predict(model, x):
    return model.predict(np.asarray(x, dtype="float32"), verbose=0)


class TestDiagonalShape(unittest.TestCase):

    def test_output_shape_is_units(self):
        model, _ = _diagonal(3, 6)
        self.assertEqual(_predict(model, [[1, 2, 3, 4, 5, 6]]).shape, (1, 3))

    def test_one_kernel_weight_per_input_feature(self):
        _, layer = _diagonal(3, 6)
        self.assertEqual(layer.get_weights()[0].shape, (6,))

    def test_bias_has_one_weight_per_unit(self):
        _, layer = _diagonal(3, 6)
        self.assertEqual(layer.get_weights()[1].shape, (3,))

    def test_no_bias_weight_when_use_bias_false(self):
        _, layer = _diagonal(3, 6, use_bias=False)
        self.assertEqual(len(layer.get_weights()), 1)

    def test_kernel_shape_is_dense_equivalent(self):
        _, layer = _diagonal(3, 6)
        self.assertEqual(layer.kernel_shape, (6, 3))

    def test_parameter_count_is_linear_not_quadratic(self):
        _, layer = _diagonal(3, 6)
        self.assertEqual(sum(w.size for w in layer.get_weights()), 6 + 3)

    def test_compute_output_shape(self):
        _, layer = _diagonal(3, 6)
        self.assertEqual(tuple(layer.compute_output_shape((None, 6))), (None, 3))

    def test_batch_dimension_preserved(self):
        model, _ = _diagonal(2, 4)
        self.assertEqual(_predict(model, np.ones((7, 4))).shape, (7, 2))


class TestDiagonalGrouping(unittest.TestCase):
    """Each unit must read exactly one contiguous block of input_dim/units features."""

    def test_inputs_per_node_is_the_view_count(self):
        _, layer = _diagonal(3, 6)
        self.assertEqual(layer.n_inputs_per_node, 2)

    def test_sums_contiguous_blocks(self):
        model, _ = _diagonal(3, 6)
        np.testing.assert_allclose(_predict(model, [[0, 1, 2, 3, 4, 5]]), [[1, 5, 9]])

    def test_nonzero_ind_maps_each_feature_to_its_block(self):
        _, layer = _diagonal(3, 6)
        np.testing.assert_array_equal(
            layer.nonzero_ind, [[0, 0], [1, 0], [2, 1], [3, 1], [4, 2], [5, 2]])

    def test_unit_ignores_features_outside_its_block(self):
        model, layer = _diagonal(3, 6, use_bias=False)
        layer.set_weights([np.ones(6, dtype="float32")])
        base = _predict(model, [[0, 0, 0, 0, 0, 0]])
        perturbed = _predict(model, [[9, 9, 0, 0, 0, 0]])
        np.testing.assert_allclose(perturbed[0, 1:], base[0, 1:])
        self.assertNotAlmostEqual(perturbed[0, 0], base[0, 0])

    def test_each_weight_scales_only_its_own_feature(self):
        model, layer = _diagonal(3, 6, use_bias=False)
        w = np.zeros(6, dtype="float32")
        w[3] = 2.0
        layer.set_weights([w])
        np.testing.assert_allclose(_predict(model, [[1, 1, 1, 1, 1, 1]]), [[0, 2, 0]])

    def test_single_view_is_identity_on_features(self):
        model, layer = _diagonal(4, 4, use_bias=False)
        layer.set_weights([np.ones(4, dtype="float32")])
        np.testing.assert_allclose(_predict(model, [[1, 2, 3, 4]]), [[1, 2, 3, 4]])


class TestDiagonalArithmetic(unittest.TestCase):

    def test_bias_is_added_per_unit(self):
        model, layer = _diagonal(3, 6)
        layer.set_weights([np.zeros(6, dtype="float32"),
                           np.array([1.0, 2.0, 3.0], dtype="float32")])
        np.testing.assert_allclose(_predict(model, [[1, 1, 1, 1, 1, 1]]), [[1, 2, 3]])

    def test_activation_applied_after_bias(self):
        import tensorflow as tf
        model, layer = _diagonal(2, 4, activation=tf.keras.activations.get("relu"))
        layer.set_weights([np.ones(4, dtype="float32"),
                           np.array([0.0, -100.0], dtype="float32")])
        out = _predict(model, [[1, 1, 1, 1]])
        self.assertGreater(out[0, 0], 0.0)
        self.assertEqual(out[0, 1], 0.0)

    def test_zero_input_gives_bias_only(self):
        model, layer = _diagonal(2, 4)
        layer.set_weights([np.arange(4, dtype="float32"),
                           np.array([5.0, 7.0], dtype="float32")])
        np.testing.assert_allclose(_predict(model, [[0, 0, 0, 0]]), [[5, 7]])

    def test_linear_in_the_input(self):
        model, _ = _diagonal(2, 4, use_bias=False)
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype="float32")
        np.testing.assert_allclose(_predict(model, 2 * x), 2 * _predict(model, x), rtol=1e-6)


class TestDiagonalDivisibility(unittest.TestCase):
    """
    input_dim/units is the per-gene view count. A non-integral ratio means the
    feature matrix is not laid out in equal per-gene blocks, and the reshape in
    call() would silently regroup features across gene boundaries.
    """

    def test_non_divisible_input_is_rejected(self):
        with self.assertRaises(ValueError):
            _diagonal(3, 7)

    def test_error_names_the_offending_dimensions(self):
        with self.assertRaises(ValueError) as ctx:
            _diagonal(3, 7)
        self.assertIn("7", str(ctx.exception))
        self.assertIn("3", str(ctx.exception))

    def test_divisible_input_still_builds(self):
        model, _ = _diagonal(3, 9)
        self.assertEqual(_predict(model, np.ones((1, 9))).shape, (1, 3))

    def test_inputs_per_node_is_an_integer(self):
        _, layer = _diagonal(3, 9)
        self.assertIsInstance(layer.n_inputs_per_node, int)


class TestSparseTF(unittest.TestCase):

    def setUp(self):
        # a -> p, b -> q, c -> both
        self.map = pd.DataFrame([[1, 0], [0, 1], [1, 1]],
                                index=["a", "b", "c"], columns=["p", "q"]).astype(float)

    def test_output_shape_is_units(self):
        model, _ = _sparse(self.map)
        self.assertEqual(_predict(model, [[1, 2, 3]]).shape, (1, 2))

    def test_one_weight_per_nonzero_map_entry(self):
        _, layer = _sparse(self.map)
        self.assertEqual(layer.get_weights()[0].shape, (int(self.map.values.sum()),))

    def test_weight_count_is_below_the_dense_equivalent(self):
        _, layer = _sparse(self.map)
        self.assertLess(layer.get_weights()[0].size, self.map.shape[0] * self.map.shape[1])

    def test_nonzero_ind_matches_the_map(self):
        _, layer = _sparse(self.map)
        expected = np.array(np.nonzero(self.map.values)).T
        np.testing.assert_array_equal(layer.nonzero_ind, expected)

    def test_kernel_shape_is_dense_equivalent(self):
        _, layer = _sparse(self.map)
        self.assertEqual(layer.kernel_shape, (3, 2))

    def test_computes_masked_dot_product(self):
        model, _ = _sparse(self.map)
        np.testing.assert_allclose(_predict(model, [[1, 2, 3]]), [[4, 5]])

    def test_unconnected_input_cannot_influence_an_output(self):
        """The biological constraint: no edge in the map, no path in the network."""
        model, _ = _sparse(self.map)
        base = _predict(model, [[0, 0, 0]])
        # 'b' connects only to q, so p must be unchanged
        bumped = _predict(model, [[0, 5, 0]])
        self.assertAlmostEqual(bumped[0, 0], base[0, 0])
        self.assertNotAlmostEqual(bumped[0, 1], base[0, 1])

    def test_shared_input_reaches_every_connected_output(self):
        model, _ = _sparse(self.map)
        base = _predict(model, [[0, 0, 0]])
        bumped = _predict(model, [[0, 0, 5]])
        self.assertNotAlmostEqual(bumped[0, 0], base[0, 0])
        self.assertNotAlmostEqual(bumped[0, 1], base[0, 1])

    def test_all_zero_map_column_gives_constant_output(self):
        m = pd.DataFrame([[1, 0], [1, 0]], index=["a", "b"], columns=["p", "q"]).astype(float)
        model, _ = _sparse(m)
        np.testing.assert_allclose(_predict(model, [[3, 4]])[:, 1], [0.0])

    def test_bias_added_when_requested(self):
        model, layer = _sparse(self.map, use_bias=True)
        weights = layer.get_weights()
        layer.set_weights([np.zeros_like(weights[0]),
                           np.array([1.0, 2.0], dtype="float32")])
        np.testing.assert_allclose(_predict(model, [[1, 2, 3]]), [[1, 2]])

    def test_no_bias_weight_by_default(self):
        _, layer = _sparse(self.map)
        self.assertEqual(len(layer.get_weights()), 1)

    def test_activation_applied(self):
        import tensorflow as tf
        model, _ = _sparse(self.map, activation=tf.keras.activations.get("relu"))
        np.testing.assert_allclose(_predict(model, [[-1, -2, -3]]), [[0, 0]])

    def test_compute_output_shape(self):
        _, layer = _sparse(self.map)
        self.assertEqual(tuple(layer.compute_output_shape((None, 3))), (None, 2))

    def test_linear_in_the_input(self):
        model, _ = _sparse(self.map)
        x = np.array([[1.0, 2.0, 3.0]], dtype="float32")
        np.testing.assert_allclose(_predict(model, 2 * x), 2 * _predict(model, x), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
