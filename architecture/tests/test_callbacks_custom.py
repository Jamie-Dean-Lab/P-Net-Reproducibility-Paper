import math
import unittest
from unittest.mock import MagicMock, patch

from architecture.callbacks_custom import TQDMCallback, step_decay

INIT_LR = 1e-3
DROP = 0.25
EPOCHS_DROP = 50


def _lr(epoch, init_lr=INIT_LR, drop=DROP, epochs_drop=EPOCHS_DROP):
    return step_decay(epoch, init_lr, drop, epochs_drop)


class TestStepDecay(unittest.TestCase):

    def test_matches_the_reference_formula(self):
        for epoch in (0, 1, 49, 50, 137, 299):
            expected = INIT_LR * math.pow(DROP, math.floor((1 + epoch) / EPOCHS_DROP))
            self.assertAlmostEqual(_lr(epoch), expected, places=15)

    def test_starts_at_the_initial_rate(self):
        self.assertAlmostEqual(_lr(0), INIT_LR, places=15)

    def test_constant_within_a_step(self):
        self.assertEqual({_lr(e) for e in range(0, 49)}, {INIT_LR})

    def test_drops_once_the_step_boundary_is_crossed(self):
        self.assertAlmostEqual(_lr(49), INIT_LR * DROP, places=15)

    def test_boundary_is_at_epoch_plus_one(self):
        """floor((1 + epoch) / epochs_drop) means the drop lands on epoch 49, not 50."""
        self.assertAlmostEqual(_lr(48), INIT_LR, places=15)
        self.assertLess(_lr(49), _lr(48))

    def test_each_step_multiplies_by_drop(self):
        self.assertAlmostEqual(_lr(99), _lr(49) * DROP, places=15)
        self.assertAlmostEqual(_lr(149), _lr(99) * DROP, places=15)

    def test_never_increases(self):
        rates = [_lr(e) for e in range(300)]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_stays_positive_over_a_full_run(self):
        self.assertGreater(_lr(299), 0.0)

    def test_drop_of_one_keeps_the_rate_flat(self):
        self.assertEqual({_lr(e, drop=1.0) for e in range(200)}, {INIT_LR})

    def test_scales_linearly_with_initial_rate(self):
        self.assertAlmostEqual(_lr(137, init_lr=2e-3), 2 * _lr(137), places=15)

    def test_smaller_epochs_drop_decays_faster(self):
        self.assertLess(_lr(100, epochs_drop=10), _lr(100, epochs_drop=100))

    def test_returns_a_float(self):
        self.assertIsInstance(_lr(10), float)


class TestTQDMCallback(unittest.TestCase):

    def setUp(self):
        self.patcher = patch("architecture.callbacks_custom.tqdm")
        self.tqdm = self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.bar = MagicMock()
        self.tqdm.return_value = self.bar
        self.cb = TQDMCallback(300)

    def test_bar_sized_to_total_epochs(self):
        self.cb.on_train_begin()
        self.tqdm.assert_called_once_with(total=300)

    def test_advances_once_per_epoch(self):
        self.cb.on_train_begin()
        for epoch in range(3):
            self.cb.on_epoch_end(epoch)
        self.assertEqual(self.bar.update.call_count, 3)

    def test_description_is_one_indexed(self):
        self.cb.on_train_begin()
        self.cb.on_epoch_end(0)
        self.bar.set_description.assert_called_with("Epoch 1/300")

    def test_bar_closed_at_end_of_training(self):
        self.cb.on_train_begin()
        self.cb.on_train_end()
        self.bar.close.assert_called_once()

    def test_no_bar_before_training_starts(self):
        self.assertIsNone(TQDMCallback(10).progress_bar)


if __name__ == "__main__":
    unittest.main()
