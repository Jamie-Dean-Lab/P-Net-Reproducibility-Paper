import logging, os
import numpy as np
from architecture.pnet_model import TFModel

class Pipeline:
    """
    Base class to setup the general structure for experiments
    """
    def __init__(self, config : dict):
        """
        Initialise the pipeline with a config file. See config_templates.py for examples
        """
        # Keep config reference
        self.config = config
    
    def _sanitise_config(self, inp):
        """
        Tries to convert the config file into a json compatible string. Serialisation is problematic
        due to functions / classes
        """
        terms = []
        if type(inp) is dict:
            for k,v in inp.items():
                if type(v) == int or type(v) == float:
                    terms.append(f'"{k}" : {v}')
                elif type(v) == str:
                    terms.append(f'"{k}" : "{v}"')
                elif type(v) == tuple or type(v) == list:
                    terms.append(f'"{k}" : [' + self._sanitise_config(v) + "]")
                elif type(v) == dict:
                    terms.append(f'"{k}" : ' + "{" + self._sanitise_config(v) + "}")
                else:
                    try:
                        v = str(float(v))
                    except:
                        v = str(v).replace("\n", "").replace("\"", "")
                    terms.append(f'"{k}" : "{v}"')
        elif type(inp) is list or type(inp) is tuple:
            for v in inp:
                if type(v) == int or type(v) == float:
                    terms.append(str(v))
                elif type(v) == str:
                    terms.append(f'"{v}"')
                elif type(v) == tuple or type(v) == list:
                    terms.append("[" + self._sanitise_config(v) + "]")
                elif type(v) == dict:
                    terms.append("{" + self._sanitise_config(v) + "}")
                else:
                    try:
                        v = str(float(v))
                    except:
                        v = str(v).replace("\n", "").replace("\"", "")
                    terms.append(f'"{v}"')
        return ",".join(terms)
    
    def run_single_split(self, load_data=True):
        """
        Runs a single split pipeline with the currently loaded config.

        args:
            load_data (bool) : Determines if data should be loaded/reloaded on this run
        """
        # Set up directory structure and apparatus for logging the run
        self.run_dir = os.path.join(self.config["run_dir"], self.config["run_id"])
        self.log = self._get_logger("main_logger", self.run_dir)
        self.log.info("Beginning run {}".format(self.config["run_id"]))
        self.log.info("Configuration file used : {}".format(self.config))
        self.fold_logger = self.log
        # Load in the data
        if load_data:
            # To save time don't reload data if same data is going to be reused
            self._load_data()
        self._summarise_data()
        # Perform training and evaluation
        test_dir = os.path.join(self.config["run_dir"], self.config["run_id"])
        if "grid_search" not in self.config.keys():
            # Get splits
            train_df, val_df, test_df = self.data.get_specific_split(self.config["train_samples"],
                                                                        self.config["val_samples"],
                                                                        self.config["test_samples"],
                                                                        self.config["tt_split_seed"])
            self.log.info("Number of train samples : {}".format(len(train_df)))
            self.log.info("Number of validation samples : {}".format(len(val_df)))
            self.log.info("Number of test samples : {}".format(len(test_df)))
            self._fold_run(test_dir, train_df, val_df, test_df)
        else:
            for i in range(len(self.config["grid_search"])):
                # Set the config params based on grid search
                for k,v in self.config["grid_search"][i].items():
                    # Ensure not to override own grid search
                    if k != "grid_search":
                        self.config[k] = v
                # Create folder for cv runs
                gs_dir = "{}/cv_{}".format(test_dir, i)
                if not os.path.exists(gs_dir):
                    os.mkdir(gs_dir)
                # Save configuration file for this cv
                with open(f"{gs_dir}/config.txt", "w") as f:
                    f.write("{" + self._sanitise_config(self.config) + "}")
                # Prepare logging for current folder
                self.fold_logger = self._get_logger("fold_logger", gs_dir)
                # Perform training
                train_df, val_df, test_df = self.data.get_specific_split(self.config["train_samples"],
                                                                        self.config["val_samples"],
                                                                        self.config["test_samples"],
                                                                        self.config["tt_split_seed"])
                self.log.info("Number of train samples : {}".format(len(train_df)))
                self.log.info("Number of validation samples : {}".format(len(val_df)))
                self.log.info("Number of test samples : {}".format(len(test_df)))
                self._fold_run(gs_dir, train_df, val_df, test_df)

        

    def run_crossvalidation(self, load_data=True):
        """
        Runs a crossvalidation pipeline with the currently loaded config.

        args:
            load_data (bool) : Determines if data should be loaded/reloaded on this run
        """
        # Set up directory structure and apparatus for logging the run
        self.run_dir = os.path.join(self.config["run_dir"], self.config["run_id"])
        self.log = self._get_logger("main_logger", self.run_dir)
        self.log.info("Beginning run {}".format(self.config["run_id"]))
        self.log.info("Configuration file used : {}".format(self.config))
        # Load in the data
        if load_data:
            # To save time don't reload data if same data is going to be reused
            self._load_data()
        self._summarise_data()
        # Split the data into outer_kfolds train test sets
        outer_folds = self.data.get_k_splits(self.config["outer_kfolds"], self.config["tt_split_seed"])
        # Outer loop of nested crossvalidation
        for i, (train_df, test_df) in enumerate(outer_folds):
            self.log.info("Number of train samples : {}".format(len(train_df)))
            self.log.info("Number of test samples : {}".format(len(test_df)))

            self.log.info("Performing {} folds of crossvalidation on test fold {}".format(self.config["inner_kfolds"], i))

            # Create folder for test fold
            test_dir = "{}/test_{}".format(self.run_dir, i)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            # If there are no grid search params then we just default to the current settings
            if len(self.config["grid_search"]) == 0:
                self.config["grid_search"] = [self.config.copy()]
            gs_dirs = []
            for j in range(len(self.config["grid_search"])):
                # Set the config params based on grid search
                for k,v in self.config["grid_search"][j].items():
                    # Ensure not to override own grid search
                    if k != "grid_search":
                        self.config[k] = v
                # Create folder for cv runs
                gs_dir = "{}/cv_{}".format(test_dir, j)
                if not os.path.exists(gs_dir):
                    os.mkdir(gs_dir)
                gs_dirs.append(gs_dir)
                # Save configuration file for this cv
                with open(f"{gs_dir}/config.txt", "w") as f:
                    f.write("{" + self._sanitise_config(self.config) + "}")
                if self.config["inner_kfolds"] > 1:
                    # Get folds
                    folds = train_df.get_k_splits(self.config["inner_kfolds"], self.config["tv_split_seed"])
                    # Evaluate across K folds
                    fold_dirs = []
                    for k, (train_fold, val_fold) in enumerate(folds):
                        # Prepare logging for current fold
                        fold_dir = "{}/fold_{}".format(gs_dir, k)
                        if not os.path.exists(fold_dir):
                            os.mkdir(fold_dir)
                        self.fold_logger = self._get_logger("fold_logger", fold_dir)
                        # Perform fold training
                        self._fold_run(fold_dir, train_fold, val_fold, test_df._copy(range(len(test_df))))
                        fold_dirs.append(fold_dir)
                    # Collate results across folds
                    for fold_collator in self.config["fold_collators"]:
                        fold_collator({"results" : fold_dirs, "save_dir" : gs_dir})
                else:
                    # No K folds so treat it as just train test split with validation split if provided
                    train_fold, val_fold = train_df.get_train_test_split(1-self.config["validation_prop"],
                                                                        self.config["tv_split_seed"])
                    # Prepare logging for current folder
                    self.fold_logger = self._get_logger("fold_logger", gs_dir)
                    # Perform training
                    self._fold_run(gs_dir, train_fold, val_fold, test_df._copy(range(len(test_df))))

    def _get_logger(self, logger_name, log_dir):
        """
        Checks if run directory exists and if not creates it. Initialises logging to file and
        to console
        """
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        else:
            print("Directory {} already exists, overriding may occur".format(log_dir))
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        fileHandler = logging.FileHandler("{}/{}.log".format(log_dir, "run"))
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.INFO)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        log = logging.getLogger(logger_name)
        log.setLevel(logging.INFO)
        log.addHandler(fileHandler)
        log.addHandler(consoleHandler)
        return log

    def _load_data(self):
        """
        Loads in the data specified in config file
        """
        self.log.info("Loading data")
        # Instantiate the particular type of dataset to use
        self.data = self.config["dataloader"]()
        # Load in the individual view files
        view_aligner = {}
        for info in self.config["views"]:
            view_name, data_fn, selected_columns, id_col, preprocessor, aligner = info
            self.data.load_data_view(view_name, os.path.join(self.config["data_dir"], data_fn), 
                                     selected_columns, id_col, preprocessor)
            view_aligner[view_name] = aligner
        # Load in the label files
        for label_fn, id_col in self.config["labels"]:
            self.data.load_data_label(os.path.join(self.config["data_dir"], label_fn), id_col)

        # Align views
        self.data.align_views(self.config["view_alignment_method"], view_aligner)

    def _summarise_data(self):
        self.log.info("Total number of samples {}".format(len(self.data)))
        for k,v in self.data.data_views.items():
            self.log.info("View {} has {} features".format(k, v.shape[1]))
        self.log.info("{} features aligned across all views".format(len(set(self.data.get_alignment_ids()))))
        self.log.info("{} types of labels".format(self.data.labels.shape[1]))
    
    def _train(self, train_df, val_df):
        """
        Placeholder function to be overriden by subclasses for different types of model training
        """
        pass

    def _fold_run(self, fold_dir, train_fold, val_fold, test_fold):
        """
        Executes the actual training and evaluation runs. Applies data augmentation, feature selection,
        feature transformation as per the current config. Computes results on train, validation, test,
        as specified in config and runs post-processing steps as specified in config.

        args:
            fold_dir (str) : 
        """
        # Set rng seeds and try to make everything as deterministic as possible
        self.fold_logger.info("Number of samples in training folds : {}".format(len(train_fold)))
        self.fold_logger.info("Number of samples in validation fold : {}".format(len(val_fold)))
        # Perform feature selection step by fold
        feature_selector = self.config["feature_selector"]
        # Set the features for each fold
        train_fold = feature_selector.fit_transform(train_fold)
        train_fold = self.config["data_augmentor"](train_fold)
        if len(val_fold) > 0:
            val_fold = feature_selector.transform(val_fold)
        if len(test_fold) > 0:
            test_fold = feature_selector.transform(test_fold)
        self.fold_logger.info("Number of selected features : {}".format(len(train_fold.get_features())))
        # Apply preprocessing
        preprocessor = self.config["feature_preprocessor"]
        train_fold = preprocessor.fit_transform(train_fold)
        val_fold = preprocessor.transform(val_fold)
        test_fold = preprocessor.transform(test_fold)
        # Train model and save results
        self.fold_logger.info("Training model")
        model, train_hx = self._train(train_fold, val_fold)
        train_preds = model.predict(train_fold.xs)
        val_preds = model.predict(val_fold.xs) if len(val_fold) > 0 else None
        test_preds = model.predict(test_fold.xs) if len(test_fold) > 0 else None
        # Save both xs and ys so that self-supervised and semi-supervised methods can be evaluated as well
        results = {"train_preds" : train_preds, "val_preds" : val_preds, "test_preds" : test_preds,
                "train_df" : train_fold, "val_df" : val_fold, "test_df" : test_fold, "train_hx" : train_hx,
                "save_dir" : fold_dir, "model" : model, "feature_preprocessor" : preprocessor,
                "feature_selector" : feature_selector}
        # Process results as specified
        self.fold_logger.info("Saving results")
        for result_processor in self.config["results_processors"]:
            result_processor(results)
        self.fold_logger.handlers.clear()

class TFPipeline(Pipeline):
    """
    Trains a TensorFlow model
    """
    def __init__(self, config : dict):
        super().__init__(config)
        self.nn_model = TFModel(self.config["run_id"], self.config["model"], self.config["model_params"],
                                self.config["fitting_params"], self.config["feature_importance"])
    
    def _train(self, train_df, val_df):
        self.nn_model.set_params(self.config["run_id"], self.config["model"], self.config["model_params"],
                                self.config["fitting_params"], self.config["feature_importance"])
        model, train_hx = self.nn_model.fit(train_df, val_df, self.config["rng_seed"])
        return model, train_hx

    
class MLPipeline(Pipeline):
    def __init__(self, config : dict):
        super().__init__(config)

    def _train(self, train_df, val_df):
        """
        Trains a traditional ML model e.g SK Learn model. Doesn't use validation data for
        training.

        args:
            train_df (MultiViewDataset) : Dataset containing the training data
            val_df (MultiViewDataset) : Dataset containing validation data
        
        returns:
            (BaseEstimator, None) : Tuple of the fitted model and None as there is no training
                                    history
        """
        np.random.seed(self.config["rng_seed"])
        model = self.config["model"](**self.config["model_params"])
        model.fit(train_df.xs, train_df.ys)
        return model, None

def construct_gs_params(params):
    cur_param = list(params.keys())[0]
    cur_vals = params.pop(cur_param)
    if len(params) == 0:
        out = [{cur_param : v, f"{cur_param}_choice" : k} for k, v in cur_vals.items()]
        return out
    else:
        out = construct_gs_params(params)
        return [gs_param | {cur_param : v, f"{cur_param}_choice" : k} for k, v in cur_vals.items() for gs_param in out]
    
class IdentityProcessor:
    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
    
    def fit(self, dataset):
        pass

    def transform(self, dataset):
        return dataset