import os
import sys

from glioma_prediction.configs import decision_tree
from glioma_prediction.configs.decision_tree import decision_tree_config

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from glioma_prediction.configs.pnet import pnet_config
from glioma_prediction.configs.dense import dense_config
from architecture.train import train


def run():
    configs = [
        decision_tree_config,
        #pnet_config,
        #dense_config,
    ]

    for config in configs:
        train(config)


if __name__ == "__main__":
    run()
