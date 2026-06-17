import copy
import random

from .pnet_single_split_elmarakeby import pnet_single_split_elmarakeby_config

# Test-set stability: keep the model, network ordering (map_seed) and all
# hyperparameters fixed, and vary ONLY the train/val/test partition via
# tt_split_seed. Each run is therefore evaluated on a different, randomly drawn
# test set, so any variation in feature importance is attributable to the choice
# of test set alone.
n_seeds = 100
seeds = []
_seen = {42}  # exclude the default/fixed seed used elsewhere
_rng = random.Random(20240617)
while len(seeds) < n_seeds:
    candidate = _rng.randint(1, 100_000_000)
    if candidate not in _seen:
        _seen.add(candidate)
        seeds.append(candidate)

_train_prop = 0.7
_val_prop = 0.2
_test_prop = 0.1

pnet_test_set_stability_configs = [
    {
        **copy.deepcopy(pnet_single_split_elmarakeby_config),
        "run_id":          f"pnet_test_set_stability_{i}",
        "run_method":      "run_single_split",
        # Proportional split; tt_split_seed selects the random test set per run.
        "train_samples":   _train_prop,
        "val_samples":     _val_prop,
        "test_samples":    _test_prop,
        "tt_split_seed":   seed,
    }
    for i, seed in enumerate(seeds)
]
