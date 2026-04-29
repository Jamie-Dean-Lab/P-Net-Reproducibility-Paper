import unittest
import numpy as np
import pandas as pd
import os
import tempfile

from data_utils import MultiViewDataset, ConcatMultiViewDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(df: pd.DataFrame, path: str):
    """Write a DataFrame to a CSV file."""
    df.to_csv(path, index=True)


def _make_view_csv(tmp_dir, filename, ids, features, seed=0):
    """Create a temporary CSV file representing one data view."""
    rng = np.random.default_rng(seed)
    data = rng.random((len(ids), len(features))).astype(np.float32)
    df = pd.DataFrame(data, index=ids, columns=features)
    df.index.name = "id"
    path = os.path.join(tmp_dir, filename)
    _write_csv(df, path)
    return path


def _make_label_csv(tmp_dir, filename, ids, labels, seed=1):
    """Create a temporary CSV file representing labels."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 2, size=(len(ids), len(labels))).astype(np.float32)
    df = pd.DataFrame(data, index=ids, columns=labels)
    df.index.name = "id"
    path = os.path.join(tmp_dir, filename)
    _write_csv(df, path)
    return path


# ---------------------------------------------------------------------------
# Tests for MultiViewDataset
# ---------------------------------------------------------------------------

class TestMultiViewDatasetInit(unittest.TestCase):
    def test_empty_on_construction(self):
        ds = MultiViewDataset()
        self.assertEqual(ds.ids, [])
        self.assertEqual(ds.data_views, {})
        self.assertTrue(ds.labels.empty)


class TestLoadDataView(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ids = ["s1", "s2", "s3", "s4", "s5"]
        self.features = ["f1", "f2", "f3"]
        self.path = _make_view_csv(self.tmp, "view1.csv", self.ids, self.features)

    def test_load_first_view(self):
        ds = MultiViewDataset()
        out = ds.load_data_view("view1", self.path)
        self.assertIn("view1", ds.data_views)
        self.assertEqual(sorted(ds.ids), sorted(self.ids))
        self.assertEqual(ds.data_views["view1"].shape, (5, 3))

    def test_load_view_creates_correct_features(self):
        ds = MultiViewDataset()
        ds.load_data_view("view1", self.path)
        self.assertEqual(sorted(ds.data_views["view1"].columns.tolist()), sorted(self.features))

    def test_load_view_return_structure(self):
        ds = MultiViewDataset()
        out = ds.load_data_view("view1", self.path)
        for key in ("common_ids", "new_ids", "common_features", "old_features", "new_features", "old_ids"):
            self.assertIn(key, out)

    def test_load_second_view_disjoint_ids(self):
        """Loading a view with completely new IDs adds rows to both views."""
        ds = MultiViewDataset()
        ds.load_data_view("view1", self.path)

        ids2 = ["s6", "s7"]
        path2 = _make_view_csv(self.tmp, "view2.csv", ids2, ["g1", "g2"])
        ds.load_data_view("view2", path2)

        all_ids = sorted(self.ids + ids2)
        self.assertEqual(sorted(ds.ids), all_ids)
        # view1 should have NaN rows for s6, s7
        self.assertTrue(ds.data_views["view1"].loc["s6"].isna().all())

    def test_load_view_with_selected_columns(self):
        ds = MultiViewDataset()
        out = ds.load_data_view("view1", self.path, selected_columns=["f1", "f2"])
        self.assertEqual(sorted(ds.data_views["view1"].columns.tolist()), ["f1", "f2"])

    def test_load_same_view_twice_merges(self):
        """Re-loading the same view name should update existing data."""
        ds = MultiViewDataset()
        ds.load_data_view("view1", self.path)

        # Create a second file for the same view name with overlapping ids / new feature
        path2 = _make_view_csv(self.tmp, "view1b.csv", self.ids, ["f4", "f5"], seed=5)
        ds.load_data_view("view1", path2)
        cols = ds.data_views["view1"].columns.tolist()
        self.assertIn("f4", cols)
        self.assertIn("f1", cols)  # original feature still present

    def test_preprocess_applied(self):
        """Custom preprocessing function is applied to the DataFrame."""
        ds = MultiViewDataset()
        ds.load_data_view("view1", self.path, preprocess=lambda df: df * 0)
        self.assertTrue((ds.data_views["view1"].fillna(0) == 0).all().all())


class TestLoadDataLabel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ids = ["s1", "s2", "s3"]
        self.view_path = _make_view_csv(self.tmp, "view.csv", self.ids, ["f1", "f2"])
        self.label_path = _make_label_csv(self.tmp, "labels.csv", self.ids, ["label_a"])

    def test_load_label(self):
        ds = MultiViewDataset()
        ds.load_data_view("view1", self.view_path)
        ds.load_data_label(self.label_path)
        self.assertIn("label_a", ds.labels.columns)
        self.assertEqual(ds.labels.shape[0], len(self.ids))

    def test_load_label_return_structure(self):
        ds = MultiViewDataset()
        ds.load_data_view("view1", self.view_path)
        out = ds.load_data_label(self.label_path)
        for key in ("common_ids", "new_ids", "common_labels", "old_labels", "new_labels", "old_ids"):
            self.assertIn(key, out)

    def test_load_label_new_ids_added(self):
        """Labels with IDs not seen in any view still expand ds.ids."""
        ds = MultiViewDataset()
        extra_label_path = _make_label_csv(self.tmp, "labels_extra.csv", ["s99"], ["label_x"])
        ds.load_data_label(extra_label_path)
        self.assertIn("s99", ds.ids)


class TestAlignIndices(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_new_ids_padded_with_nan(self):
        ds = MultiViewDataset()
        path1 = _make_view_csv(self.tmp, "v1.csv", ["s1", "s2"], ["f1"])
        path2 = _make_view_csv(self.tmp, "v2.csv", ["s3", "s4"], ["g1"])
        ds.load_data_view("v1", path1)
        ds.load_data_view("v2", path2)
        # v1 must have NaN for s3, s4
        self.assertTrue(ds.data_views["v1"].loc["s3"].isna().all())
        self.assertTrue(ds.data_views["v1"].loc["s4"].isna().all())

    def test_ids_sorted(self):
        ds = MultiViewDataset()
        path = _make_view_csv(self.tmp, "v.csv", ["s3", "s1", "s2"], ["f1"])
        ds.load_data_view("v", path)
        self.assertEqual(ds.ids, sorted(ds.ids))


class TestGetLen(unittest.TestCase):
    def test_len(self):
        ds = MultiViewDataset()
        tmp = tempfile.mkdtemp()
        path = _make_view_csv(tmp, "v.csv", ["s1", "s2", "s3"], ["f1"])
        ds.load_data_view("v", path)
        self.assertEqual(len(ds), 3)


class TestGetItem(unittest.TestCase):
    def test_getitem_returns_dict_and_series(self):
        ds = MultiViewDataset()
        tmp = tempfile.mkdtemp()
        path = _make_view_csv(tmp, "v.csv", ["s1", "s2"], ["f1", "f2"])
        ds.load_data_view("v", path)
        xs, ys = ds[0]
        self.assertIn("v", xs)

    def test_get_views_returns_view_names(self):
        ds = MultiViewDataset()
        tmp = tempfile.mkdtemp()
        path = _make_view_csv(tmp, "v.csv", ["s1", "s2"], ["f1", "f2"])
        ds.load_data_view("my_view", path)
        views = ds.get_views()
        self.assertIsInstance(views, list)
        self.assertIn("my_view", views)


# ---------------------------------------------------------------------------
# Tests for ConcatMultiViewDataset
# ---------------------------------------------------------------------------

class TestConcatMultiViewDatasetAlignViews(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ids = ["s1", "s2", "s3", "s4", "s5", "s6"]
        self.path_v1 = _make_view_csv(self.tmp, "v1.csv", self.ids, ["g1", "g2", "g3"], seed=0)
        self.path_v2 = _make_view_csv(self.tmp, "v2.csv", self.ids, ["g2", "g3", "g4"], seed=1)
        # Binary labels (one hot style)
        labels = pd.DataFrame({"cls_0": [1, 0, 1, 0, 1, 0], "cls_1": [0, 1, 0, 1, 0, 1]},
                              index=self.ids)
        labels.index.name = "id"
        self.label_path = os.path.join(self.tmp, "labels.csv")
        _write_csv(labels, self.label_path)

    def _build(self, method="zero fill"):
        ds = ConcatMultiViewDataset()
        ds.load_data_view("v1", self.path_v1)
        ds.load_data_view("v2", self.path_v2)
        ds.load_data_label(self.label_path)
        ds.align_views(method=method)
        return ds

    def test_xs_shape_after_align(self):
        ds = self._build()
        # 2 views × 4 unique genes (g1,g2,g3,g4 padded across views)
        self.assertEqual(ds.xs.shape[1], 8)

    def test_ys_shape_after_align(self):
        ds = self._build()
        self.assertEqual(ds.ys.shape[1], 2)

    def test_zero_fill_no_nan(self):
        ds = self._build(method="zero fill")
        self.assertFalse(np.isnan(ds.xs).any())

    def test_features_list_length_matches_xs(self):
        ds = self._build()
        self.assertEqual(len(ds.features), ds.xs.shape[1])

    def test_alignment_ids_length_matches_xs(self):
        ds = self._build()
        self.assertEqual(len(ds.alignment_ids), ds.xs.shape[1])

    def test_drop_features_removes_nan_columns(self):
        # Introduce NaN by having views with different sample sets
        path_v1 = _make_view_csv(self.tmp, "vna1.csv", self.ids[:4], ["g1", "g2"], seed=10)
        path_v2 = _make_view_csv(self.tmp, "vna2.csv", self.ids[2:], ["g3", "g4"], seed=11)
        labels = pd.DataFrame({"cls_0": [1]*6, "cls_1": [0]*6}, index=self.ids)
        labels.index.name = "id"
        lpath = os.path.join(self.tmp, "lna.csv")
        _write_csv(labels, lpath)

        ds = ConcatMultiViewDataset()
        ds.load_data_view("v1", path_v1)
        ds.load_data_view("v2", path_v2)
        ds.load_data_label(lpath)
        ds.align_views(method="drop features", drop_labels=False)
        self.assertFalse(np.isnan(ds.xs).any())

    def test_get_features_returns_list(self):
        ds = self._build()
        feats = ds.get_features()
        self.assertIsInstance(feats, list)
        self.assertGreater(len(feats), 0)

    def test_get_alignment_ids_returns_list(self):
        ds = self._build()
        aids = ds.get_alignment_ids()
        self.assertIsInstance(aids, list)

    def test_getitem_returns_arrays(self):
        ds = self._build()
        x, y = ds[0]
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)


class TestAlignViewsMethods(unittest.TestCase):
    """
    Exhaustive tests for every branch of align_views:
      - method: "zero fill", "drop samples", "drop features"
      - drop_labels: True / False

    Setup creates two views with a partial-overlap in sample IDs so that
    NaN values exist in xs before any NA-handling method is applied, giving
    each branch something real to act on.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # 8 samples total; view1 covers s0-s5, view2 covers s3-s7
        # -> s0-s2 have NaN in view2 columns, s6-s7 have NaN in view1 columns
        all_ids = [f"s{i}" for i in range(8)]
        view1_ids = [f"s{i}" for i in range(6)]    # s0..s5
        view2_ids = [f"s{i}" for i in range(3, 8)] # s3..s7

        self.path_v1 = _make_view_csv(self.tmp, "v1.csv", view1_ids, ["g1", "g2"], seed=0)
        self.path_v2 = _make_view_csv(self.tmp, "v2.csv", view2_ids, ["g3", "g4"], seed=1)

        # Labels: all 8 samples, binary one-hot
        labels = pd.DataFrame(
            {"cls_0": [1, 0, 1, 0, 1, 0, 1, 0],
             "cls_1": [0, 1, 0, 1, 0, 1, 0, 1]},
            index=all_ids,
        )
        labels.index.name = "id"
        self.label_path = os.path.join(self.tmp, "labels.csv")
        _write_csv(labels, self.label_path)

        # Labels with NaN for s0 (used in drop_labels tests)
        partial_labels = labels.copy().astype(float)
        partial_labels.iloc[0] = np.nan
        self.partial_label_path = os.path.join(self.tmp, "partial_labels.csv")
        _write_csv(partial_labels, self.partial_label_path)

        self.all_ids = all_ids

    def _build(self, method, label_path=None, drop_labels=True):
        ds = ConcatMultiViewDataset()
        ds.load_data_view("v1", self.path_v1)
        ds.load_data_view("v2", self.path_v2)
        ds.load_data_label(label_path or self.label_path)
        ds.align_views(method=method, drop_labels=drop_labels)
        return ds

    # ------------------------------------------------------------------
    # zero fill
    # ------------------------------------------------------------------

    def test_zero_fill_no_nan_in_xs(self):
        ds = self._build("zero fill")
        self.assertFalse(np.isnan(ds.xs).any())

    def test_zero_fill_retains_all_samples(self):
        ds = self._build("zero fill")
        self.assertEqual(ds.xs.shape[0], 8)

    def test_zero_fill_retains_all_features(self):
        ds = self._build("zero fill")
        self.assertEqual(ds.xs.shape[1], 8)

    # ------------------------------------------------------------------
    # drop samples
    # ------------------------------------------------------------------

    def test_drop_samples_no_nan_in_xs(self):
        ds = self._build("drop samples")
        self.assertFalse(np.isnan(ds.xs).any())

    def test_drop_samples_removes_incomplete_rows(self):
        ds = self._build("drop samples")
        self.assertEqual(ds.xs.shape[0], 3)

    def test_drop_samples_ids_consistent_with_xs(self):
        ds = self._build("drop samples")
        self.assertEqual(len(ds.ids), ds.xs.shape[0])

    def test_drop_samples_ys_consistent(self):
        ds = self._build("drop samples")
        self.assertEqual(ds.ys.shape[0], ds.xs.shape[0])

    def test_drop_samples_feature_count_unchanged(self):
        ds = self._build("drop samples")
        self.assertEqual(ds.xs.shape[1], 8)

    # ------------------------------------------------------------------
    # drop features
    # ------------------------------------------------------------------

    def test_drop_features_no_nan_in_xs(self):
        ds = self._build("drop features")
        self.assertFalse(np.isnan(ds.xs).any())

    def test_drop_features_retains_all_samples(self):
        ds = self._build("drop features")
        self.assertEqual(ds.xs.shape[0], 8)

    def test_drop_features_removes_incomplete_columns(self):
        ds = self._build("drop features")
        self.assertEqual(np.isnan(ds.xs).sum(axis=0).max(), 0)

    def test_drop_features_features_list_matches_xs(self):
        ds = self._build("drop features")
        self.assertEqual(len(ds.features), ds.xs.shape[1])

    def test_drop_features_alignment_ids_matches_xs(self):
        ds = self._build("drop features")
        self.assertEqual(len(ds.alignment_ids), ds.xs.shape[1])

    # ------------------------------------------------------------------
    # drop_labels=True (default)
    # ------------------------------------------------------------------

    def test_drop_labels_true_removes_samples_with_nan_labels(self):
        ds = self._build("zero fill", label_path=self.partial_label_path, drop_labels=True)
        self.assertNotIn("s0", ds.ids)

    def test_drop_labels_true_no_nan_in_ys(self):
        ds = self._build("zero fill", label_path=self.partial_label_path, drop_labels=True)
        self.assertFalse(np.isnan(ds.ys).any())

    def test_drop_labels_true_removes_zero_label_columns(self):
        """Label columns that are all-zero after sample filtering are dropped."""
        all_ids = self.all_ids
        labels = pd.DataFrame(
            {"cls_rare":   [1, 0, 0, 0, 0, 0, 0, 0],
             "cls_common": [0, 1, 1, 1, 1, 1, 1, 1]},
            index=all_ids,
        )
        labels = labels.astype(float)
        labels.index.name = "id"
        labels.iloc[0] = np.nan
        lpath = os.path.join(self.tmp, "labels_rare.csv")
        _write_csv(labels, lpath)

        ds = self._build("zero fill", label_path=lpath, drop_labels=True)
        self.assertNotIn("cls_rare", ds.labels.columns)

    def test_drop_labels_true_xs_and_ys_row_count_match(self):
        ds = self._build("zero fill", label_path=self.partial_label_path, drop_labels=True)
        self.assertEqual(ds.xs.shape[0], ds.ys.shape[0])

    # ------------------------------------------------------------------
    # drop_labels=False
    # ------------------------------------------------------------------

    def test_drop_labels_false_retains_samples_with_nan_labels(self):
        ds = self._build("zero fill", label_path=self.partial_label_path, drop_labels=False)
        self.assertIn("s0", ds.ids)

    def test_drop_labels_false_ys_may_contain_nan(self):
        ds = self._build("zero fill", label_path=self.partial_label_path, drop_labels=False)
        self.assertTrue(np.isnan(ds.ys).any())

    def test_drop_labels_false_never_reduces_below_drop_true(self):
        ds_keep = self._build("zero fill", drop_labels=False)
        ds_drop = self._build("zero fill", drop_labels=True)
        self.assertGreaterEqual(ds_keep.xs.shape[0], ds_drop.xs.shape[0])


class TestConcatMultiViewDatasetSplits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        n = 20
        self.ids = [f"s{i}" for i in range(n)]
        path_v = _make_view_csv(self.tmp, "v.csv", self.ids, ["g1", "g2", "g3"], seed=0)
        labels = pd.DataFrame(
            {"cls_0": ([1, 0] * (n // 2)), "cls_1": ([0, 1] * (n // 2))},
            index=self.ids
        )
        labels.index.name = "id"
        lpath = os.path.join(self.tmp, "labels.csv")
        _write_csv(labels, lpath)

        self.ds = ConcatMultiViewDataset()
        self.ds.load_data_view("v", path_v)
        self.ds.load_data_label(lpath)
        self.ds.align_views(method="zero fill")

    # --- Train / test split ---

    def test_train_test_split_sizes(self):
        train, test = self.ds.get_train_test_split(0.8)
        total = len(train) + len(test)
        self.assertEqual(total, len(self.ds))

    def test_train_test_split_no_overlap(self):
        train, test = self.ds.get_train_test_split(0.8)
        train_ids = set(train.ids)
        test_ids = set(test.ids)
        self.assertEqual(len(train_ids & test_ids), 0)

    def test_train_test_split_stratified(self):
        train, test = self.ds.get_train_test_split(0.8, stratified=True)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)

    def test_train_test_split_xs_shape(self):
        train, test = self.ds.get_train_test_split(0.8)
        self.assertEqual(train.xs.shape[1], self.ds.xs.shape[1])
        self.assertEqual(test.xs.shape[1], self.ds.xs.shape[1])

    def test_train_test_stratified_preserves_class_proportions(self):
        """Stratified split should preserve class proportions within tolerance."""
        train, test = self.ds.get_train_test_split(0.8, stratified=True)

        overall_pos_rate = self.ds.ys[:, 0].mean()
        train_pos_rate = train.ys[:, 0].mean()
        test_pos_rate = test.ys[:, 0].mean()

        self.assertAlmostEqual(train_pos_rate, overall_pos_rate, delta=0.05,
            msg=f"Train class proportion {train_pos_rate:.2f} deviates from "
                f"overall {overall_pos_rate:.2f}")
        self.assertAlmostEqual(test_pos_rate, overall_pos_rate, delta=0.05,
            msg=f"Test class proportion {test_pos_rate:.2f} deviates from "
                f"overall {overall_pos_rate:.2f}")

    # --- K-fold splits ---

    def test_k_splits_count(self):
        folds = list(self.ds.get_k_splits(5))
        self.assertEqual(len(folds), 5)

    def test_k_splits_no_overlap(self):
        for train, val in self.ds.get_k_splits(4):
            self.assertEqual(len(set(train.ids) & set(val.ids)), 0)

    def test_k_splits_cover_all_samples(self):
        """All samples appear in a validation set exactly once across all folds."""
        seen = set()
        for _, val in self.ds.get_k_splits(5):
            seen.update(val.ids)
        self.assertEqual(seen, set(self.ds.ids))

    def test_k_splits_stratified(self):
        folds = list(self.ds.get_k_splits(4, stratified=True))
        self.assertEqual(len(folds), 4)

    def test_k_splits_stratified_preserves_class_proportions(self):
        """Each validation fold should have approximately the same class proportion as the full dataset."""
        overall_pos_rate = self.ds.ys[:, 0].mean()

        for i, (train, val) in enumerate(self.ds.get_k_splits(4, stratified=True)):
            val_pos_rate = val.ys[:, 0].mean()
            self.assertAlmostEqual(val_pos_rate, overall_pos_rate, delta=0.1,
                msg=f"Fold {i} val class proportion {val_pos_rate:.2f} deviates from "
                    f"overall {overall_pos_rate:.2f}")

    # --- Specific split ---

    def test_specific_split_by_ids(self):
        train_ids = self.ids[:10]
        val_ids = self.ids[10:15]
        test_ids = self.ids[15:]
        train, val, test = self.ds.get_specific_split(train_ids, val_ids, test_ids)
        self.assertEqual(sorted(train.ids), sorted(train_ids))
        self.assertEqual(sorted(val.ids), sorted(val_ids))
        self.assertEqual(sorted(test.ids), sorted(test_ids))

    def test_specific_split_by_proportion(self):
        train, val, test = self.ds.get_specific_split(0.6, 0.2, 0.2)
        total = len(train) + len(val) + len(test)
        self.assertLessEqual(total, len(self.ds))

    def test_specific_split_no_overlap_ids(self):
        train, val, test = self.ds.get_specific_split(0.6, 0.2, 0.2)
        s_train = set(train.ids)
        s_val = set(val.ids)
        s_test = set(test.ids)
        self.assertEqual(len(s_train & s_val), 0)
        self.assertEqual(len(s_train & s_test), 0)
        self.assertEqual(len(s_val & s_test), 0)

    def test_specific_split_oversized_test_raises(self):
        with self.assertRaises(Exception):
            self.ds.get_specific_split(0.5, 0.3, 1.5)


class TestConcatMultiViewDatasetCopy(unittest.TestCase):
    """Internal _copy() should produce an independent deep copy of the subset."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        ids = [f"s{i}" for i in range(10)]
        path_v = _make_view_csv(self.tmp, "v.csv", ids, ["g1", "g2"], seed=0)
        labels = pd.DataFrame({"cls_0": [1, 0] * 5, "cls_1": [0, 1] * 5}, index=ids)
        labels.index.name = "id"
        lpath = os.path.join(self.tmp, "labels.csv")
        _write_csv(labels, lpath)
        self.ds = ConcatMultiViewDataset()
        self.ds.load_data_view("v", path_v)
        self.ds.load_data_label(lpath)
        self.ds.align_views(method="zero fill")

    def test_copy_subset_ids(self):
        copy = self.ds._copy([0, 1, 2])
        self.assertEqual(len(copy.ids), 3)

    def test_copy_xs_shape(self):
        copy = self.ds._copy([0, 1, 2])
        self.assertEqual(copy.xs.shape, (3, self.ds.xs.shape[1]))

    def test_copy_independent(self):
        copy = self.ds._copy([0, 1, 2])
        copy.xs[0, 0] = -9999
        self.assertNotEqual(self.ds.xs[0, 0], -9999)


# ---------------------------------------------------------------------------
# load_data_view — untested argument and return-value paths
# ---------------------------------------------------------------------------

class TestLoadDataViewEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ids = ["s1", "s2", "s3"]
        self.features = ["f1", "f2", "f3"]
        self.path = _make_view_csv(self.tmp, "view.csv", self.ids, self.features)

    def test_selected_columns_nonexistent_ignored(self):
        """Columns that don't exist in the file are silently dropped via intersection."""
        ds = MultiViewDataset()
        ds.load_data_view("v", self.path, selected_columns=["f1", "DOES_NOT_EXIST"])
        self.assertIn("f1", ds.data_views["v"].columns)
        self.assertNotIn("DOES_NOT_EXIST", ds.data_views["v"].columns)

    def test_selected_columns_all_nonexistent_loads_empty_features(self):
        """If none of the selected columns exist the view should have zero feature columns."""
        ds = MultiViewDataset()
        ds.load_data_view("v", self.path, selected_columns=["DOES_NOT_EXIST"])
        self.assertEqual(ds.data_views["v"].shape[1], 0)

    def test_id_col_nonzero(self):
        """id_col parameter selects the correct column as the index."""
        df = pd.DataFrame({
            "numeric_extra": [9.0, 8.0, 7.0],
            "sample_id":     ["s1", "s2", "s3"],
            "feat":          [1.0, 2.0, 3.0],
        })
        path = os.path.join(self.tmp, "nonzero_id.csv")
        df.to_csv(path, index=False)
        ds = MultiViewDataset()
        ds.load_data_view("v", path, id_col=1)
        self.assertEqual(sorted(ds.ids), ["s1", "s2", "s3"])

    def test_return_common_ids_populated(self):
        """common_ids should list IDs that already existed when the same view is reloaded."""
        ds = MultiViewDataset()
        ds.load_data_view("v", self.path)
        out = ds.load_data_view("v", self.path)
        self.assertEqual(sorted(out["common_ids"]), sorted(self.ids))

    def test_return_old_ids_populated(self):
        """old_ids should list IDs present before but absent from the new file."""
        ds = MultiViewDataset()
        ds.load_data_view("v", self.path)
        path2 = _make_view_csv(self.tmp, "partial.csv", ["s1"], ["f1"])
        out = ds.load_data_view("v", path2)
        self.assertIn("s2", out["old_ids"])
        self.assertIn("s3", out["old_ids"])

    def test_return_common_features_populated(self):
        """common_features lists features present in both the old and new load."""
        ds = MultiViewDataset()
        ds.load_data_view("v", self.path)
        path2 = _make_view_csv(self.tmp, "overlap.csv", self.ids, ["f2", "f3", "f4"])
        out = ds.load_data_view("v", path2)
        self.assertIn("f2", out["common_features"])
        self.assertIn("f3", out["common_features"])

    def test_return_old_features_populated(self):
        """old_features lists features in the existing view that the new file lacks."""
        ds = MultiViewDataset()
        ds.load_data_view("v", self.path)
        path2 = _make_view_csv(self.tmp, "new_only.csv", self.ids, ["f4"])
        out = ds.load_data_view("v", path2)
        self.assertIn("f1", out["old_features"])


# ---------------------------------------------------------------------------
# load_data_label — untested argument and return-value paths
# ---------------------------------------------------------------------------

class TestLoadDataLabelEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ids = ["s1", "s2", "s3"]
        self.view_path = _make_view_csv(self.tmp, "view.csv", self.ids, ["f1"])

    def test_id_col_nonzero_label(self):
        """id_col parameter selects the correct column as the label index."""
        df = pd.DataFrame({
            "extra": ["x", "y", "z"],
            "sample_id": ["s1", "s2", "s3"],
            "label": [1.0, 0.0, 1.0],
        })
        path = os.path.join(self.tmp, "nonzero_id_label.csv")
        df.to_csv(path, index=False)
        ds = MultiViewDataset()
        ds.load_data_view("v", self.view_path)
        ds.load_data_label(path, id_col=1)
        self.assertEqual(sorted(ds.labels.index.tolist()), sorted(self.ids))

    def test_load_second_label_file_adds_new_column(self):
        """Loading a second label file with a new column name extends the label DataFrame."""
        label_path1 = _make_label_csv(self.tmp, "lab1.csv", self.ids, ["label_a"])
        label_path2 = _make_label_csv(self.tmp, "lab2.csv", self.ids, ["label_b"])
        ds = MultiViewDataset()
        ds.load_data_view("v", self.view_path)
        ds.load_data_label(label_path1)
        ds.load_data_label(label_path2)
        self.assertIn("label_a", ds.labels.columns)
        self.assertIn("label_b", ds.labels.columns)

    def test_load_second_label_file_updates_common_labels(self):
        """Reloading a label column with new values overwrites the existing values."""
        df1 = pd.DataFrame({"label_a": [0.0, 0.0, 0.0]}, index=self.ids)
        df1.index.name = "id"
        path1 = os.path.join(self.tmp, "lab_zeros.csv")
        _write_csv(df1, path1)
        df2 = pd.DataFrame({"label_a": [1.0, 1.0, 1.0]}, index=self.ids)
        df2.index.name = "id"
        path2 = os.path.join(self.tmp, "lab_ones.csv")
        _write_csv(df2, path2)

        ds = MultiViewDataset()
        ds.load_data_view("v", self.view_path)
        ds.load_data_label(path1)
        ds.load_data_label(path2)
        self.assertTrue((ds.labels["label_a"] == 1.0).all())

    def test_get_labels_returns_label_names(self):
        label_path = _make_label_csv(self.tmp, "lab.csv", self.ids, ["label_a", "label_b"])
        ds = MultiViewDataset()
        ds.load_data_view("v", self.view_path)
        ds.load_data_label(label_path)
        labels = ds.get_labels()
        self.assertIsInstance(labels, list)
        self.assertIn("label_a", labels)
        self.assertIn("label_b", labels)

    def test_load_label_return_common_ids(self):
        label_path = _make_label_csv(self.tmp, "lab.csv", self.ids, ["label_a"])
        ds = MultiViewDataset()
        ds.load_data_view("v", self.view_path)
        ds.load_data_label(label_path)
        out = ds.load_data_label(label_path)
        self.assertEqual(sorted(out["common_ids"]), sorted(self.ids))

    def test_load_label_return_new_labels(self):
        path1 = _make_label_csv(self.tmp, "la.csv", self.ids, ["label_a"])
        path2 = _make_label_csv(self.tmp, "lb.csv", self.ids, ["label_b"])
        ds = MultiViewDataset()
        ds.load_data_view("v", self.view_path)
        ds.load_data_label(path1)
        out = ds.load_data_label(path2)
        self.assertIn("label_b", out["new_labels"])

    def test_load_label_return_old_labels(self):
        path1 = _make_label_csv(self.tmp, "la.csv", self.ids, ["label_a"])
        path2 = _make_label_csv(self.tmp, "lb.csv", self.ids, ["label_b"])
        ds = MultiViewDataset()
        ds.load_data_view("v", self.view_path)
        ds.load_data_label(path1)
        out = ds.load_data_label(path2)
        self.assertIn("label_a", out["old_labels"])


# ---------------------------------------------------------------------------
# Stratified splits — single-label (binary scalar) branch
# ---------------------------------------------------------------------------

class TestStratifiedSingleLabel(unittest.TestCase):
    """
    _get_train_test_split and _get_k_splits each have two stratification paths:
      (a) multi-column one-hot  -> labels.shape[1] > 1
      (b) single binary column  -> labels.shape[1] == 1
    All existing stratified tests use (a). These tests exercise (b).
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        n = 20
        self.ids = [f"s{i}" for i in range(n)]
        path_v = _make_view_csv(self.tmp, "v.csv", self.ids, ["g1", "g2"], seed=0)
        labels = pd.DataFrame(
            {"binary": ([0, 1] * (n // 2))},
            index=self.ids,
        )
        labels.index.name = "id"
        lpath = os.path.join(self.tmp, "labels_single.csv")
        _write_csv(labels, lpath)

        self.ds = ConcatMultiViewDataset()
        self.ds.load_data_view("v", path_v)
        self.ds.load_data_label(lpath)
        self.ds.align_views(method="zero fill")

    def test_train_test_stratified_single_label_sizes(self):
        train, test = self.ds.get_train_test_split(0.8, stratified=True)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertEqual(len(train) + len(test), len(self.ds))

    def test_train_test_stratified_single_label_no_overlap(self):
        train, test = self.ds.get_train_test_split(0.8, stratified=True)
        self.assertEqual(len(set(train.ids) & set(test.ids)), 0)

    def test_train_test_stratified_single_label_preserves_class_proportions(self):
        """Stratified split on a single binary label should preserve class proportions."""
        train, test = self.ds.get_train_test_split(0.8, stratified=True)

        overall_pos_rate = self.ds.ys[:, 0].mean()
        train_pos_rate = train.ys[:, 0].mean()
        test_pos_rate = test.ys[:, 0].mean()

        self.assertAlmostEqual(train_pos_rate, overall_pos_rate, delta=0.05,
            msg=f"Train class proportion {train_pos_rate:.2f} deviates from "
                f"overall {overall_pos_rate:.2f}")
        self.assertAlmostEqual(test_pos_rate, overall_pos_rate, delta=0.05,
            msg=f"Test class proportion {test_pos_rate:.2f} deviates from "
                f"overall {overall_pos_rate:.2f}")

    def test_k_splits_stratified_single_label_count(self):
        folds = list(self.ds.get_k_splits(4, stratified=True))
        self.assertEqual(len(folds), 4)

    def test_k_splits_stratified_single_label_no_overlap(self):
        for train, val in self.ds.get_k_splits(4, stratified=True):
            self.assertEqual(len(set(train.ids) & set(val.ids)), 0)

    def test_k_splits_stratified_single_label_preserves_class_proportions(self):
        """Each validation fold should preserve class proportions for a single binary label."""
        overall_pos_rate = self.ds.ys[:, 0].mean()

        for i, (train, val) in enumerate(self.ds.get_k_splits(4, stratified=True)):
            val_pos_rate = val.ys[:, 0].mean()
            self.assertAlmostEqual(val_pos_rate, overall_pos_rate, delta=0.1,
                msg=f"Fold {i} val class proportion {val_pos_rate:.2f} deviates from "
                    f"overall {overall_pos_rate:.2f}")


# ---------------------------------------------------------------------------
# _get_k_splits — n_splits=1 early-return path
# ---------------------------------------------------------------------------

class TestKSplitsEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        ids = [f"s{i}" for i in range(10)]
        path_v = _make_view_csv(self.tmp, "v.csv", ids, ["g1"], seed=0)
        labels = pd.DataFrame({"cls_0": [1, 0] * 5, "cls_1": [0, 1] * 5}, index=ids)
        labels.index.name = "id"
        lpath = os.path.join(self.tmp, "labels.csv")
        _write_csv(labels, lpath)
        self.ds = ConcatMultiViewDataset()
        self.ds.load_data_view("v", path_v)
        self.ds.load_data_label(lpath)
        self.ds.align_views(method="zero fill")

    def test_k_splits_one_fold_returns_all_train_empty_val(self):
        folds = list(self.ds.get_k_splits(1))
        self.assertEqual(len(folds), 1)
        train, val = folds[0]
        self.assertEqual(len(train.ids), len(self.ds))
        self.assertEqual(len(val.ids), 0)


# ---------------------------------------------------------------------------
# _get_specific_split — oversized validation and train proportion errors
# ---------------------------------------------------------------------------

class TestSpecificSplitErrors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        ids = [f"s{i}" for i in range(10)]
        path_v = _make_view_csv(self.tmp, "v.csv", ids, ["g1"], seed=0)
        labels = pd.DataFrame({"cls_0": [1, 0] * 5, "cls_1": [0, 1] * 5}, index=ids)
        labels.index.name = "id"
        lpath = os.path.join(self.tmp, "labels.csv")
        _write_csv(labels, lpath)
        self.ds = ConcatMultiViewDataset()
        self.ds.load_data_view("v", path_v)
        self.ds.load_data_label(lpath)
        self.ds.align_views(method="zero fill")

    def test_oversized_val_proportion_raises(self):
        with self.assertRaises(Exception):
            self.ds.get_specific_split(0.5, 1.5, 0.1)

    def test_oversized_train_proportion_raises(self):
        with self.assertRaises(Exception):
            self.ds.get_specific_split(1.5, 0.1, 0.1)

    def test_mixed_list_and_proportion_splits(self):
        """test as explicit list, val and train as proportions."""
        test_ids = [f"s{i}" for i in range(3)]
        train, val, test = self.ds.get_specific_split(0.5, 0.2, test_ids)
        self.assertEqual(sorted(test.ids), sorted(test_ids))
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertEqual(len(set(train.ids) & set(test.ids)), 0)
        self.assertEqual(len(set(val.ids) & set(test.ids)), 0)


# ---------------------------------------------------------------------------
# align_views — view_aligner custom mapping path
# ---------------------------------------------------------------------------

class TestAlignViewsViewAligner(unittest.TestCase):
    """
    The view_aligner dict lets callers supply a function per view that maps
    raw column names to canonical alignment IDs.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        ids = [f"s{i}" for i in range(6)]
        self.path_expr   = _make_view_csv(self.tmp, "expr.csv",   ids, ["ENSG001", "ENSG002"], seed=0)
        self.path_methyl = _make_view_csv(self.tmp, "methyl.csv", ids, ["ENSG001_me", "ENSG002_me"], seed=1)
        labels = pd.DataFrame({"cls_0": [1, 0] * 3, "cls_1": [0, 1] * 3}, index=ids)
        labels.index.name = "id"
        self.label_path = os.path.join(self.tmp, "labels.csv")
        _write_csv(labels, self.label_path)

    def test_view_aligner_produces_correct_alignment_ids(self):
        ds = ConcatMultiViewDataset()
        ds.load_data_view("expr",   self.path_expr)
        ds.load_data_view("methyl", self.path_methyl)
        ds.load_data_label(self.label_path)
        ds.align_views(
            method="zero fill",
            view_aligner={"methyl": lambda col: col.replace("_me", "")},
        )
        self.assertIn("ENSG001", ds.alignment_ids)
        self.assertIn("ENSG002", ds.alignment_ids)

    def test_view_aligner_interleaves_views_by_gene(self):
        """Columns from each view for the same gene should appear contiguously."""
        ds = ConcatMultiViewDataset()
        ds.load_data_view("expr",   self.path_expr)
        ds.load_data_view("methyl", self.path_methyl)
        ds.load_data_label(self.label_path)
        ds.align_views(
            method="zero fill",
            view_aligner={"methyl": lambda col: col.replace("_me", "")},
        )
        ensg001_positions = [i for i, aid in enumerate(ds.alignment_ids) if aid == "ENSG001"]
        ensg002_positions = [i for i, aid in enumerate(ds.alignment_ids) if aid == "ENSG002"]
        self.assertEqual(max(ensg001_positions) - min(ensg001_positions), 1)
        self.assertEqual(max(ensg002_positions) - min(ensg002_positions), 1)

    def test_view_aligner_alignment_ids_length_matches_features(self):
        ds = ConcatMultiViewDataset()
        ds.load_data_view("expr",   self.path_expr)
        ds.load_data_view("methyl", self.path_methyl)
        ds.load_data_label(self.label_path)
        ds.align_views(
            method="zero fill",
            view_aligner={"methyl": lambda col: col.replace("_me", "")},
        )
        self.assertEqual(len(ds.alignment_ids), len(ds.features))
        self.assertEqual(len(ds.alignment_ids), ds.xs.shape[1])


if __name__ == "__main__":
    unittest.main()