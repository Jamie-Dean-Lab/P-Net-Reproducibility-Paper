from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

class FigureROCConfiguration(Enum):
    plot_size = (7, 7)
    plot_colors = ['magenta', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'yellow',
                   'cyan', 'black', 'gold', 'teal']
    top_spine_visibility = False
    bottom_spine_visibility = True
    left_spine_visibility = True
    right_spine_visibility = False
    x_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    x_axis_limit = (0, 1.06)
    y_axis_limit = (0, 1.06)
    spine_thickness = 1
    tick_size = 12
    label_size = 12
    legend_size = 10


class PlotROC:

    def __init__(self, results):
        self.results = results
        self.config = FigureROCConfiguration

    def process_results(self):
        self.processed_results = {}
        for model_name, model_results in self.results.items():
            self.processed_results[model_name] = {
                'y_true': np.array(model_results['response']),
                'pred_scores': np.array(model_results['response_pred'])
            }

    def _format_axes(self, ax):
        ax.spines['right'].set_visible(self.config.right_spine_visibility.value)
        ax.spines['left'].set_visible(self.config.left_spine_visibility.value)
        ax.spines['top'].set_visible(self.config.top_spine_visibility.value)
        ax.spines['bottom'].set_visible(self.config.bottom_spine_visibility.value)

        ax.set_xticks(self.config.x_ticks.value)
        ax.set_yticks(self.config.y_ticks.value)
        ax.set_xlim(*self.config.x_axis_limit.value)
        ax.set_ylim(*self.config.y_axis_limit.value)

        ax.spines['left'].set_linewidth(self.config.spine_thickness.value)
        ax.spines['bottom'].set_linewidth(self.config.spine_thickness.value)

        remove_zero = lambda x, _: '' if x == 0 else x
        ax.xaxis.set_major_formatter(plt.FuncFormatter(remove_zero))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(remove_zero))

        ax.tick_params(axis='x', labelsize=self.config.tick_size.value)
        ax.tick_params(axis='y', labelsize=self.config.tick_size.value)

        ax.set_xlabel('FPR', fontsize=self.config.label_size.value)
        ax.set_ylabel('TPR', fontsize=self.config.label_size.value)
        return ax

    def plot(self, ax, title):
        self.process_results()
        ax.set_title(title, loc="left", fontdict={"fontsize": 14, "fontweight": "bold"})

        # random baseline
        ax.plot([0, 1], [0, 1], linestyle='--', color='grey', alpha=0.5)

        for i, (model_name, model_results) in enumerate(self.processed_results.items()):
            y_true = model_results['y_true']
            pred_scores = model_results['pred_scores']

            fpr, tpr, _ = roc_curve(y_true, pred_scores)
            auroc = roc_auc_score(y_true, pred_scores)
            label = f'{model_name} ({round(auroc, 2)})'
            ax.plot(fpr, tpr, label=label, c=self.config.plot_colors.value[i])

        ax = self._format_axes(ax)
        ax.legend(loc='lower right', fontsize=self.config.legend_size.value)