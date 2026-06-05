import unittest
import pandas as pd
import numpy as np

from prostate_cancer_prediction.feature_encoders import mut_binary, cnv_del, cnv_amp, cnv, cnv_signed


class TestMutBinary(unittest.TestCase):

    def test_values_above_one_set_to_one(self):
        df = pd.DataFrame({"gene1": [2.0, 3.0, 100.0], "gene2": [5.0, 10.0, 2.0]})
        result = mut_binary(df)
        self.assertEqual((result > 1.0).sum().sum(), 0)

    def test_values_of_one_unchanged(self):
        df = pd.DataFrame({"gene1": [1.0, 1.0], "gene2": [1.0, 1.0]})
        result = mut_binary(df)
        self.assertTrue((result == 1.0).all().all())

    def test_values_of_zero_unchanged(self):
        df = pd.DataFrame({"gene1": [0.0, 0.0], "gene2": [0.0, 0.0]})
        result = mut_binary(df)
        self.assertTrue((result == 0.0).all().all())

    def test_mixed_values(self):
        df = pd.DataFrame({"gene1": [0.0, 1.0, 2.0, 5.0]})
        result = mut_binary(df)
        expected = pd.DataFrame({"gene1": [0.0, 1.0, 1.0, 1.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_negative_values_unchanged(self):
        df = pd.DataFrame({"gene1": [-1.0, -2.0]})
        result = mut_binary(df)
        pd.testing.assert_frame_equal(result, pd.DataFrame({"gene1": [-1.0, -2.0]}))

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = mut_binary(df)
        self.assertTrue(result.empty)

    def test_returns_dataframe(self):
        df = pd.DataFrame({"gene1": [1.0, 2.0]})
        result = mut_binary(df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_single_cell_above_one(self):
        df = pd.DataFrame({"gene1": [99.0]})
        result = mut_binary(df)
        self.assertEqual(result.iloc[0, 0], 1.0)

    def test_all_zeros(self):
        df = pd.DataFrame(np.zeros((5, 5)))
        result = mut_binary(df)
        self.assertTrue((result == 0.0).all().all())


class TestCnvDel(unittest.TestCase):

    def test_positive_values_set_to_zero(self):
        df = pd.DataFrame({"gene1": [1.0, 2.0, 3.0]})
        result = cnv_del(df)
        self.assertTrue((result == 0.0).all().all())

    def test_zero_set_to_zero(self):
        df = pd.DataFrame({"gene1": [0.0]})
        result = cnv_del(df)
        self.assertEqual(result.iloc[0, 0], 0.0)

    def test_single_deletion_ignored(self):
        df = pd.DataFrame({"gene1": [-1.0]})
        result = cnv_del(df)
        self.assertEqual(result.iloc[0, 0], 0.0)

    def test_double_deletion_becomes_one(self):
        df = pd.DataFrame({"gene1": [-2.0]})
        result = cnv_del(df)
        self.assertEqual(result.iloc[0, 0], 1.0)

    def test_mixed_values(self):
        df = pd.DataFrame({"gene1": [2.0, 0.0, -1.0, -2.0]})
        result = cnv_del(df)
        expected = pd.DataFrame({"gene1": [0.0, 0.0, 0.0, 1.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_all_amplifications_become_zero(self):
        df = pd.DataFrame({"gene1": [1.0, 2.0, 3.0, 4.0]})
        result = cnv_del(df)
        self.assertTrue((result == 0.0).all().all())

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = cnv_del(df)
        self.assertTrue(result.empty)

    def test_returns_dataframe(self):
        df = pd.DataFrame({"gene1": [-2.0]})
        result = cnv_del(df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_multi_column(self):
        df = pd.DataFrame({"gene1": [-2.0, -1.0], "gene2": [1.0, -2.0]})
        result = cnv_del(df)
        expected = pd.DataFrame({"gene1": [1.0, 0.0], "gene2": [0.0, 1.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_large_negative_values_unchanged(self):
        df = pd.DataFrame({"gene1": [-3.0, -5.0]})
        result = cnv_del(df)
        self.assertEqual(result.iloc[0, 0], -3.0)
        self.assertEqual(result.iloc[1, 0], -5.0)


class TestCnvAmp(unittest.TestCase):

    def test_negative_values_set_to_zero(self):
        df = pd.DataFrame({"gene1": [-1.0, -2.0, -3.0]})
        result = cnv_amp(df)
        self.assertTrue((result == 0.0).all().all())

    def test_zero_set_to_zero(self):
        df = pd.DataFrame({"gene1": [0.0]})
        result = cnv_amp(df)
        self.assertEqual(result.iloc[0, 0], 0.0)

    def test_single_amplification_ignored(self):
        df = pd.DataFrame({"gene1": [1.0]})
        result = cnv_amp(df)
        self.assertEqual(result.iloc[0, 0], 0.0)

    def test_double_amplification_becomes_one(self):
        df = pd.DataFrame({"gene1": [2.0]})
        result = cnv_amp(df)
        self.assertEqual(result.iloc[0, 0], 1.0)

    def test_mixed_values(self):
        df = pd.DataFrame({"gene1": [-2.0, 0.0, 1.0, 2.0]})
        result = cnv_amp(df)
        expected = pd.DataFrame({"gene1": [0.0, 0.0, 0.0, 1.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_all_deletions_become_zero(self):
        df = pd.DataFrame({"gene1": [-1.0, -2.0, -3.0, -4.0]})
        result = cnv_amp(df)
        self.assertTrue((result == 0.0).all().all())

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = cnv_amp(df)
        self.assertTrue(result.empty)

    def test_returns_dataframe(self):
        df = pd.DataFrame({"gene1": [2.0]})
        result = cnv_amp(df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_multi_column(self):
        df = pd.DataFrame({"gene1": [2.0, 1.0], "gene2": [-1.0, 2.0]})
        result = cnv_amp(df)
        expected = pd.DataFrame({"gene1": [1.0, 0.0], "gene2": [0.0, 1.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_large_positive_values_unchanged(self):
        df = pd.DataFrame({"gene1": [3.0, 5.0]})
        result = cnv_amp(df)
        self.assertEqual(result.iloc[0, 0], 3.0)
        self.assertEqual(result.iloc[1, 0], 5.0)


class TestCnv(unittest.TestCase):

    def test_single_deletion_dropped(self):
        df = pd.DataFrame({"gene1": [-1.0]})
        result = cnv(df)
        self.assertEqual(result.iloc[0, 0], 0.0)

    def test_double_deletion_becomes_minus_one(self):
        df = pd.DataFrame({"gene1": [-2.0]})
        result = cnv(df)
        self.assertEqual(result.iloc[0, 0], -1.0)

    def test_single_amplification_dropped(self):
        df = pd.DataFrame({"gene1": [1.0]})
        result = cnv(df)
        self.assertEqual(result.iloc[0, 0], 0.0)

    def test_double_amplification_becomes_one(self):
        df = pd.DataFrame({"gene1": [2.0]})
        result = cnv(df)
        self.assertEqual(result.iloc[0, 0], 1.0)

    def test_zero_unchanged(self):
        df = pd.DataFrame({"gene1": [0.0]})
        result = cnv(df)
        self.assertEqual(result.iloc[0, 0], 0.0)

    def test_mixed_values(self):
        df = pd.DataFrame({"gene1": [-2.0, -1.0, 0.0, 1.0, 2.0]})
        result = cnv(df)
        expected = pd.DataFrame({"gene1": [-1.0, 0.0, 0.0, 0.0, 1.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_out_of_range_values_unchanged(self):
        df = pd.DataFrame({"gene1": [-3.0, 3.0]})
        result = cnv(df)
        self.assertEqual(result.iloc[0, 0], -3.0)
        self.assertEqual(result.iloc[1, 0], 3.0)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = cnv(df)
        self.assertTrue(result.empty)

    def test_returns_dataframe(self):
        df = pd.DataFrame({"gene1": [2.0]})
        result = cnv(df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_multi_column(self):
        df = pd.DataFrame({"gene1": [-2.0, 1.0], "gene2": [2.0, -1.0]})
        result = cnv(df)
        expected = pd.DataFrame({"gene1": [-1.0, 0.0], "gene2": [1.0, 0.0]})
        pd.testing.assert_frame_equal(result, expected)


class TestCnvSigned(unittest.TestCase):

    def test_single_deletion_kept_as_minus_one(self):
        df = pd.DataFrame({"gene1": [-1.0]})
        result = cnv_signed(df)
        self.assertEqual(result.iloc[0, 0], -1.0)

    def test_double_deletion_clamped_to_minus_one(self):
        df = pd.DataFrame({"gene1": [-2.0]})
        result = cnv_signed(df)
        self.assertEqual(result.iloc[0, 0], -1.0)

    def test_single_amplification_kept_as_one(self):
        df = pd.DataFrame({"gene1": [1.0]})
        result = cnv_signed(df)
        self.assertEqual(result.iloc[0, 0], 1.0)

    def test_double_amplification_clamped_to_one(self):
        df = pd.DataFrame({"gene1": [2.0]})
        result = cnv_signed(df)
        self.assertEqual(result.iloc[0, 0], 1.0)

    def test_zero_unchanged(self):
        df = pd.DataFrame({"gene1": [0.0]})
        result = cnv_signed(df)
        self.assertEqual(result.iloc[0, 0], 0.0)

    def test_mixed_values(self):
        df = pd.DataFrame({"gene1": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]})
        result = cnv_signed(df)
        expected = pd.DataFrame({"gene1": [-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_output_bounded_to_signed_range(self):
        df = pd.DataFrame({"gene1": [-5.0, 4.0, 100.0]})
        result = cnv_signed(df)
        self.assertTrue((result >= -1.0).all().all())
        self.assertTrue((result <= 1.0).all().all())

    def test_single_events_retained_unlike_cnv(self):
        df = pd.DataFrame({"gene1": [-1.0, 1.0]})
        cnv_result = cnv(df.copy())
        signed_result = cnv_signed(df.copy())
        self.assertTrue((cnv_result == 0.0).all().all())
        pd.testing.assert_frame_equal(signed_result, pd.DataFrame({"gene1": [-1.0, 1.0]}))

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = cnv_signed(df)
        self.assertTrue(result.empty)

    def test_returns_dataframe(self):
        df = pd.DataFrame({"gene1": [2.0]})
        result = cnv_signed(df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_multi_column(self):
        df = pd.DataFrame({"gene1": [-2.0, 1.0], "gene2": [2.0, -1.0]})
        result = cnv_signed(df)
        expected = pd.DataFrame({"gene1": [-1.0, 1.0], "gene2": [1.0, -1.0]})
        pd.testing.assert_frame_equal(result, expected)


class TestSymmetry(unittest.TestCase):

    def test_deletion_not_amplification(self):
        df_del = pd.DataFrame({"gene1": [-2.0]})
        df_amp = pd.DataFrame({"gene1": [-2.0]})
        self.assertEqual(cnv_del(df_del).iloc[0, 0], 1.0)
        self.assertEqual(cnv_amp(df_amp).iloc[0, 0], 0.0)

    def test_amplification_not_deletion(self):
        df_del = pd.DataFrame({"gene1": [2.0]})
        df_amp = pd.DataFrame({"gene1": [2.0]})
        self.assertEqual(cnv_del(df_del).iloc[0, 0], 0.0)
        self.assertEqual(cnv_amp(df_amp).iloc[0, 0], 1.0)

    def test_no_overlap_on_full_range(self):
        values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        df_del = pd.DataFrame({"gene1": values})
        df_amp = pd.DataFrame({"gene1": values})
        result_del = cnv_del(df_del)
        result_amp = cnv_amp(df_amp)
        both_one = (result_del == 1.0) & (result_amp == 1.0)
        self.assertFalse(both_one.any().any())


class TestInputNotMutated(unittest.TestCase):

    def _assert_unchanged(self, encoder):
        df = pd.DataFrame({"gene1": [-2.0, -1.0, 0.0, 1.0, 2.0], "gene2": [2.0, 1.0, 0.0, -1.0, -2.0]})
        original = df.copy()
        encoder(df)
        pd.testing.assert_frame_equal(df, original)

    def test_mut_binary_does_not_mutate_input(self):
        self._assert_unchanged(mut_binary)

    def test_cnv_del_does_not_mutate_input(self):
        self._assert_unchanged(cnv_del)

    def test_cnv_amp_does_not_mutate_input(self):
        self._assert_unchanged(cnv_amp)

    def test_cnv_does_not_mutate_input(self):
        self._assert_unchanged(cnv)

    def test_cnv_signed_does_not_mutate_input(self):
        self._assert_unchanged(cnv_signed)

    def test_returns_new_object(self):
        df = pd.DataFrame({"gene1": [2.0, -2.0]})
        for encoder in (mut_binary, cnv_del, cnv_amp, cnv, cnv_signed):
            self.assertIsNot(encoder(df), df)


if __name__ == "__main__":
    unittest.main()