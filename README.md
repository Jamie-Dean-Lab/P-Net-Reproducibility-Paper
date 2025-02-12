# P-Net-Reproducibility-Paper

This is the implementation of P-NET (a Biologically Informed Neural Network). The code was written by Haitham A Elmarakeby et al. in “Biologically informed deep neural network for prostate cancer discover" (paper link: https://www.nature.com/articles/s41586-021-03922-4) and adapted to work for Python 3. It is used for three tasks:

Prostate Cancer Classification (primary or metastatic) - reproducibility of the results outlined in the paper
Gliom Classification (Lower Grade Glioma or Glioblastoma) - using the TCGA dataset
Radiosensitivity Prediction - using 511 cell lines from the CCLE dataset

The code has been heavily refactored to allow input datasets to be more easily integrated for experiments beyond the original Prostate Cancer Classification task. Issues around reproducibility of results have also been fixed and the pipeline and configuration has been cleaned up to ensure that settings within the configuration files will influence each run where previously sometimes there was overriding of parameters hidden in the pipeline. The pipeline has also been designed to be more extensible, allowing users to write their own functions to be inserted at different stages of the pipeline for the purposes of experiments and are described below

## Structure of the repository
All code pertaining to P-Net and supporting pipeline can be found in the architecture folder. All other folders are experiment specific and this is the intended way to use the repository.

### Architecture
1. pnet_model.py - contains the code for constructing TensorFlow implementation of P-Net from Reactome.
2. pipeline.py - contains code for the pipeline object that is used to configure and run experiments with P-Net and other models. MLPipeline is used for any sklearn type model and TFPipeline is used for P-Net but in general the Pipeline class is designed such that you can subclass it and override _train to do any platform / framework specific changes before fitting the model.
3. data_utils.py - contains code for the classes used for loading multiple views of a dataset and integrates them for use with P-Net and other models. Features from multiple views are aligned according to alignment_ids and they features of the same alignment_id are kept contiguously when concatenated together.
4. layers_custom.py - contains Diagonal and SparseTF which are mostly unchanged from the original P-Net repository
5. callbacks_custom.py - callback functions to be used with tensorflow models, mostly unchanged from the original P-Net repository
6. coef_weights_utils.py - functions to help with extracting coefficients and outputs from the tensorflow model layers for the purpose of deeplift / explainability. Mostly unchanged from original P-Net repository
7. deepexplain - folder containing deepexplain / deeplift code. Mostly unchanged from original P-Net repository
8. Reactome - folder containing the reactome data used by P-Net. Unchanged from the original P-Net repository
9. evaluation.py - contains functions that are attached to the results_processors variable in the configuration file to allow flexibility in what kinds of evaluations to perform on each run e.g AUC, accuracy, F1, train history, deeplift etc.
10. pnet_config.py - template configuration file with values set to what was specified in the P-Net paper (which is not necessarily the same as in the original P-Net github repository). Gives an idea of what is available for configuration and what are expected inputs

### Installation
