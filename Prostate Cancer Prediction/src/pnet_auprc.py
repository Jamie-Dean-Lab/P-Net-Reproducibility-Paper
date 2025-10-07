
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from enum import Enum

class DataLoaderAUPRC:

    def __init__(self, results_folder):
        self.results_folder = results_folder

    def load_data(self):
        """
        Load the results for all the trained models. To indicate a results file, it must finish
        with '_results.csv'. By convention, the first part of the file name represents the model
        name.

        Parameters
        ----------
        None

        Returns
        ----------
        results: dict
            Maps model names to dataframes representing results of evaluating the model on a
            training/test set; the dataframe must have the 'y' column indicating the ground
            truth labels and the 'pred_scores' indicating prediction probabilities for the
            positive class in the case of binary classification
        """
        results = {}
        results_files_pattern = '*test_results.csv'
        results_file_paths = glob.glob(os.path.join(self.results_folder, results_files_pattern))
        results_file_names = [os.path.basename(path) for path in results_file_paths]
        model_names = [name.split('_')[0] for name in results_file_names]
        for name, results_path in zip(model_names, results_file_paths):
            result_df = pd.read_csv(results_path)
            results[name] = result_df
        return results

class FigureAUPRCConfiguration(Enum):
    """
    Class to collect all the configurable parameters for the Area Under
    Precision Recall Curve (AUPRC) curve.
    """
    plot_size = (7, 7)

    plot_colors = ['magenta', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'yellow']

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


class PlotAUPRC:

    def __init__(self, results):
        self.results = results
        self.config = FigureAUPRCConfiguration

    def process_results(self):
        """
        Convert the raw results into processed inputs to be further used for computing
        precision, recall and auprc.
        """
        self.processed_results = {}
        for model_name, model_results in self.results.items():
            processed_model_res = {}

            # these are needed to compute the precision and recall at various thresholds
            processed_model_res['y_true'] = np.array(model_results['response'])
            processed_model_res['pred_scores'] = np.array(model_results['response_pred'])
            self.processed_results[model_name] = processed_model_res

    @staticmethod
    def _compute_auprc(recall_list, precision_list):
        """
        Given a list of computed recalls and precisions (at different thresholds),
        compute the AUPRC.
        
        Note: when using sklrean for computing precision and recall,
        the two lists are already ordered. To allow custom functions for computing the metrics,
        this function orders the precision and recall lists before computing the AUPRC.
        """
        increasing_indices = np.argsort(recall_list)
        sorted_recall_list = recall_list[increasing_indices]
        sorted_precision_list = precision_list[increasing_indices]
        auprc = auc(recall_list, precision_list) #auc(sorted_recall_list, sorted_precision_list)
        return auprc
    
    # adapted from the scikit learn tutorial: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    @staticmethod
    def _plot_f1_isocurves(ax):
        """
        Plot F1 isocurves - curves containing points for each the F1 score has
        a constant, pre-defined value.
        """
        f1_scores = np.linspace(0.2, 0.8, num=4)
        x = np.linspace(0.01, 1)

        for i, f_score in enumerate(f1_scores):
            y_raw = f_score * x / (2 * x - f_score)

            # ensure the isocurve does not exceed the boundaries of the plot
            y = y_raw[y_raw<=1.05]
            x = x[y_raw<=1.05]

            ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.3)
            annotation_label = "F1={0:0.1f}".format(f_score) if i==0 else "{0:0.1f}".format(f_score)

            # ensure labels are associated to each curve (values here are tweaked for 4 curves)
            ax.annotate(annotation_label, xy=(f1_scores[i]-0.18*(i>0)-0.15*(i==0), 1.05), fontsize=12)
        return ax
    
    def _format_axes(self, ax: plt.Axes):
        """Apply formatting on axes spines, ticks and labels."""

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

        ax.set_xlabel('Recall', fontsize=self.config.label_size.value)
        ax.set_ylabel('Precision', fontsize=self.config.label_size.value)

        return ax
    
    def plot(self, save=False, save_dir=None, show=False):
        """Plot the Area Under Precision Recall Curve."""

        # construct processed inputs
        self.process_results()

        fig, ax = plt.subplots(figsize=self.config.plot_size.value)

        for i, (model_name, model_results) in enumerate(self.processed_results.items()):
            y_true = model_results['y_true']
            pred_scores = model_results['pred_scores']

            # compute a list of (precision, recall) pairs at various thresholds
            precision_list, recall_list, _ = precision_recall_curve(y_true, pred_scores)
            auprc = average_precision_score(y_true, pred_scores)
            label = f'{model_name} ({round(auprc, 2)})'
            ax.plot(recall_list, precision_list, label=label, c=self.config.plot_colors.value[i])

        ax = self._plot_f1_isocurves(ax)
        ax = self._format_axes(ax)
        ax.legend(loc='lower left')

        if save:
            save_dir = os.getcwd() if (save_dir is None) else save_dir
            save_file_path = os.path.join(save_dir, 'area_under_precision_recall_curve.jpg')
            fig.savefig(save_file_path)
        
        if show:
            plt.show()
        else:
            plt.close()


            