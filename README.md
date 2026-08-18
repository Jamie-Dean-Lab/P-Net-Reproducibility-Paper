# P-Net-Reproducibility-Paper

This is an implementation of P-NET (a Biologically Informed Neural Network). The original code was written by Haitham A Elmarakeby et al. for "Biologically informed deep neural network for prostate cancer discovery" (paper link: https://www.nature.com/articles/s41586-021-03922-4) and has been adapted here to work with Python 3. It is used for four tasks:

1. Prostate Cancer Classification (primary or metastatic) - reproducibility of the results outlined in the paper
2. Tissue type classification - using the adult GTEx dataset
3. Radiosensitivity Prediction - using 427 cell lines from the CCLE dataset
4. Lower grade glioma vs GBM classification - using the TCGA Lower Grade Glioma and Glioblastoma (GBMLGG) dataset.

The code has been heavily refactored so that new input datasets can be integrated more easily, for experiments beyond the original Prostate Cancer Classification task. Issues around reproducibility of results have also been fixed, and the pipeline and configuration have been cleaned up so that the settings in a configuration file always take effect for that run, where previously some parameters were silently overridden inside the pipeline. The pipeline has also been made more extensible, allowing users to supply their own functions at different stages of the pipeline for the purposes of experiments; these entry points are described below.

## Installation

### Windows / Linux
Recommended to install with some kind of environment managing software like conda

```
conda create --name pnet-repro python=3.10
conda activate pnet-repro
```
Install the CUDA toolkit and cuDNN for GPU support before installing requirements
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
```
Install requirements file from the root directory of the repository
```
pip install -r requirements.txt
```

### macOS (Apple Silicon)
TensorFlow on Apple Silicon requires the `tensorflow-macos` and `tensorflow-metal` packages instead of the standard `tensorflow`. The metal plugin version must be pinned to match the TensorFlow version.

```
conda create --name pnet-repro python=3.10
conda activate pnet-repro
```
Install TensorFlow and the Metal GPU plugin separately before the rest of the requirements
```
pip install tensorflow-macos==2.10.0 tensorflow-metal==0.6.0
```
LightGBM on macOS requires `libomp`, which is not installed by default
```
brew install libomp
```
Then install the remaining requirements
```
pip install -r requirements-mac.txt
```

Kaleido (used to export Sankey diagrams as PNG) requires Google Chrome or Chromium to be installed on all platforms. If Chrome is not already installed, you can install a compatible version by running `kaleido_get_chrome` after installing the requirements.

## Running the experiments
All scripts should be run from the root directory rather than from the individual experiment folders. `run_all.py` in the root directory will execute all models across all four tasks sequentially, but this is expected to take an extremely long time. It is strongly recommended to run each task's `run.py` individually instead. If you are running the radiosensitivity experiment scripts you will also need to have R installed.

## Datasets
All datasets are downloaded automatically the first time each experiment's script is run, so no manual data download is required. The tissue type, radiosensitivity and glioma datasets are downloaded from https://zenodo.org/records/21979483, and the prostate cancer dataset is downloaded from the original P-Net release at https://zenodo.org/records/10775529. The radiosensitivity task additionally downloads the Cleveland radiosensitivity data through an R script, which is why R is required for that task.

## Structure of the repository
All code pertaining to P-Net and the supporting pipeline can be found in the architecture folder. All other folders are experiment specific and this is the intended way to use the repository.

### Architecture
1. pnet_model.py - contains the code for constructing the TensorFlow implementation of P-Net from a pathway hierarchy (Reactome by default, or GO).
2. pnet_config.py - factory for the P-Net model_params block shared by every task's configs, so that each config file only states what it actually changes.
3. pipeline.py - contains code for the pipeline object that is used to configure and run experiments with P-Net and other models. MLPipeline is used for any sklearn type model and TFPipeline is used for P-Net but in general the Pipeline class is designed such that you can subclass it and override _train to do any platform / framework specific changes before fitting the model.
4. data_utils.py - contains code for the classes used for loading multiple views of a dataset and integrating them for use with P-Net and other models. Features from multiple views are aligned according to alignment_ids, and the features sharing an alignment_id are kept contiguous when the views are concatenated together.
5. data_loading.py - single implementation of turning a config's views and labels into an aligned dataset. It is used by the pipeline and by the standalone scripts that need to reproduce a run's exact sample set or fold composition without starting a pipeline.
6. layers_custom.py - contains Diagonal and SparseTF which are mostly unchanged from the original P-Net repository
7. callbacks_custom.py - callback functions to be used with tensorflow models, mostly unchanged from the original P-Net repository
8. coef_weights_utils.py - functions to help with extracting coefficients and outputs from the tensorflow model layers for the purpose of deeplift / explainability. Mostly unchanged from original P-Net repository
9. deepexplain - folder containing deepexplain / deeplift code. Mostly unchanged from original P-Net repository
10. Reactome - folder containing the reactome data used by P-Net. Unchanged from the original P-Net repository
11. GO - folder containing a Gene Ontology hierarchy and gene sets, offering an alternative to Reactome for constructing the P-Net masks (used by the pnet_GO experiment config). preprocess.py builds the hierarchy, gene-set, and id-map files from the source CX file.
12. evaluation.py - contains functions that are attached to the results_processors variable in the configuration file to allow flexibility in what kinds of evaluations to perform on each run e.g AUC, accuracy, F1, train history, deeplift etc.
13. dense_model.py - constructs a plain fully-connected (dense) TensorFlow network used as a non-biologically-informed baseline against P-Net.
14. run_pipeline.py - small helper that takes a config, instantiates its pipeline_class, and invokes the configured run_method. This is the entry point each experiment's configs are passed through.
15. sensitivity_split.py - shared code for extracting the fixed train/test split that the hyperparameter sensitivity sweeps run on, so that they use the exact samples of a chosen fold of the main crossvalidation.
16. significance_utils.py - shared significance testing for the model comparisons, using the Nadeau & Bengio corrected resampled t-test to account for the overlapping training sets of crossvalidation folds.
17. plotting - shared figures used across the tasks (model comparison across folds, hyperparameter sensitivity, and the Sankey diagram of P-Net pathway importances).
18. tests - pytest suite covering the architecture code.

### Usage
Each experiment folder contains a `configs/` directory holding a `base_config.py` with the shared settings for that task plus one config file per model/experiment (e.g. `adaboost.py`, `pnet.py`), and a `run.py` that imports the desired configs and passes each to `architecture.run_pipeline.run_pipeline`. To set up a new experiment, copy an existing config, spread in the relevant `base_config`, and edit it to suit your needs. There are a few config options that are experiment specific and listed below
1. run_id - specifies the tag for the current experiment run
2. data_dir - path to the folder containing all the data for the experiments
3. run_dir - path to the folder you wish to store all the outputs of your experiments
4. views - list of tuples describing the datasets you wish to load in, each of the form (identifying tag for the kind of data view, filename within data_dir, the columns to select, the index column, a function to preprocess the data, and a function to extract alignment_ids from the headers)
5. view_alignment_method - a string to specify how to deal with NAs when aligning different views
6. labels - list of (filename, index column) tuples pointing at the response variables you wish to make a prediction for
   
The pipeline has 2 run methods. The first method is run_single_split. This is used when you do not want to do full crossvalidation, and lets you split the data into train, validation, and test sets either based on a random seed or on lists of sample ids for each split. To specify these splits you need to set the following config variables
1. train_samples - either a list of sample ids or a float between 0 and 1 specifying the size of the train set
2. val_samples - either a list of sample ids or a float between 0 and 1 specifying the size of the validation set
3. test_samples - either a list of sample ids or a float between 0 and 1 specifying the size of the test set
4. tt_split_seed - random seed used to make the split reproducible when proportions are given instead of sample ids
   
The second method is run_crossvalidation. For this you will need to specify a few extra config variables
1. tt_split_seed - random seed to make the outer development-test splits reproducible
2. tv_split_seed - random seed to make train-validation splits reproducible
3. inner_kfolds - number of train-validation splits to compute per test split
4. outer_kfolds - number of development-test splits for the crossvalidation
5. stratified - whether the splits should be stratified by label
6. validation_prop - a float specifying proportion of the development set to be used for validation. Only used when inner_kfolds is less than 2, or when hold_out_validation_for_final_fit is set
#### Customisable entry points
The pipeline was designed to let users customise different steps in the model development pipeline beyond specifying parameters and hyperparameters of the model.
1. feature_selector - a class that follows the same format as a sklearn model e.g with fit, fit_transform, transform methods. The purpose of this entry point is to let users define a feature selection method that is fitted on the training data of each crossvalidation fold independently, and then applied to the validation and test sets when evaluating
2. data_augmentor - a function that takes in the training dataset and outputs an augmented training dataset e.g with artificial new data points
3. results_processors - a list of functions that are run after a model for a training run has been completed. This can be various metrics, plotting training history, saving model weights etc.
   
#### Grid search
Grid searching is configured by assigning a dictionary to the `grid_search` variable in the config, where each config item that you want to gridsearch over is a key in the dictionary and the value is a dictionary of the parameters to be searched. The keys of the inner dictionary are just identifiers for that particular parameter setting and the value is the actual value you want to gridsearch over. The pipeline automatically expands this into the full grid (via `construct_gs_params`) at run time.
