import unittest
import pandas as pd
import numpy as np

from prostate_cancer_prediction.preprocess import mut_binary, cnv_del, cnv_amp


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


if __name__ == "__main__":
    unittest.main()