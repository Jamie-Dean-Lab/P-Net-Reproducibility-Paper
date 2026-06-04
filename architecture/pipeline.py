import copy
import logging, os
import numpy as np
import pandas as pd
from architecture.pnet_model import TFModel


class Pipeline:
    """
    Base class to setup the general structure for experiments
    """

    def __init__(self, config: dict):
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
            for k, v in inp.items():
                if type(v) == int or type(v) == float:
                    terms.append(f'"{k}" : {v}')
                elif type(v) == str:
                    v = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace(
                        "\t", "\\t")
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
                    v = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace(
                        "\t", "\\t")
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
        self.log.debug("Configuration file used : {}".format(self.config))
        self.fold_logger = self.log
        # Load in the data
        if load_data:
            self._load_data(self.config["shuffle_seed"])
        self._summarise_data()
        # Perform training and evaluation
        test_dir = os.path.join(self.config["run_dir"], self.config["run_id"])
        if len(self.config["grid_search"]) == 0:
            train_df, val_df, test_df = self.data.get_specific_split(self.config["train_samples"],
                                                                     self.config["val_samples"],
                                                                     self.config["test_samples"],
                                                                     self.config["tt_split_seed"])
            self.log.info("Number of train samples : {}".format(len(train_df)))
            self.log.info("Number of validation samples : {}".format(len(val_df)))
            self.log.info("Number of test samples : {}".format(len(test_df)))
            self._fold_run(test_dir, train_df, val_df, test_df)
        else:
            self.config["grid_search"] = construct_gs_params(self.config["grid_search"])
            gs_dirs = []
            gs_params = []
            training_results = []
            # Perform training
            train_df, val_df, test_df = self.data.get_specific_split(self.config["train_samples"],
                                                                     self.config["val_samples"],
                                                                     self.config["test_samples"],
                                                                     self.config["tt_split_seed"])
            for i in range(len(self.config["grid_search"])):
                for k, v in self.config["grid_search"][i].items():
                    if k != "grid_search":
                        self.config[k] = v
                gs_dir = "{}/cv_{}".format(test_dir, i)
                if not os.path.exists(gs_dir):
                    os.mkdir(gs_dir)
                with open(f"{gs_dir}/config.txt", "w") as f:
                    f.write("{" + self._sanitise_config(self.config) + "}")
                self.fold_logger = self._get_logger("fold_logger", gs_dir)
                self.log.info("Number of train samples : {}".format(len(train_df)))
                self.log.info("Number of validation samples : {}".format(len(val_df)))
                self.log.info("Number of test samples : {}".format(len(test_df)))
                val_result = self._fold_run(gs_dir, train_df, val_df, [])
                training_results.append(val_result)
            if len(self.config["val_metric"]) > 0:
                training_results = pd.DataFrame(training_results)
                for metric in training_results.columns:
                    best_p = training_results[metric].idxmax()
                    for k, v in self.config["grid_search"][best_p].items():
                        if k != "grid_search":
                            self.config[k] = v
                    gs_params.append(self.config["grid_search"][best_p]["model_params_choice"])
                    best_dir = f"{test_dir}/best_{metric}"
                    gs_dirs.append(best_dir)
                    if not os.path.exists(best_dir):
                        os.mkdir(best_dir)
                    with open(f"{best_dir}/config.txt", "w") as f:
                        f.write("{" + self._sanitise_config(self.config) + "}")
                    if self.config["hold_out_validation_for_final_fit"]:
                        self.fold_logger = self._get_logger("fold_logger", best_dir)
                        self._fold_run(best_dir, train_df, val_df, test_df)
                    else:
                        best_train_df, _, best_test_df = self.data.get_specific_split(
                            self.config["train_samples"] + self.config["val_samples"],
                            0,
                            self.config["test_samples"],
                            self.config["tt_split_seed"]
                        )
                        self.fold_logger = self._get_logger("fold_logger", best_dir)
                        self._fold_run(best_dir, best_train_df, _, best_test_df)
                for gsc in self.config["grid_search_collators"]:
                    gsc({"gs_dirs": gs_dirs, "params": gs_params, "save_dir": self.run_dir})
                if "external_datasets" in self.config:
                    self._run_external_validation()

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
        self.log.debug("Configuration file used : {}".format(self.config))
        # Load in the data
        if load_data:
            # To save time don't reload data if same data is going to be reused
            self._load_data(self.config["shuffle_seed"])
        self._summarise_data()
        if "test_samples" in self.config.keys():
            train_df, _, test_df = self.data.get_specific_split(self.config["train_samples"],
                                                                self.config["val_samples"],
                                                                self.config["test_samples"],
                                                                self.config["tt_split_seed"])
            outer_folds = [(train_df, test_df)]
        else:
            # Split the data into outer_kfolds train test sets
            if self.config["outer_kfolds"] < 2:
                raise Exception("For nested crossvalidation at least 2 outer_kfolds needed")
            outer_folds = self.data.get_k_splits(self.config["outer_kfolds"], self.config["stratified"],
                                                 self.config["tt_split_seed"])

        # Outer loop of nested crossvalidation
        gs_dirs = []
        gs_params = []
        test_dirs = []

        # If there are no grid search params then we just default to the current settings
        if len(self.config["grid_search"]) == 0:
            default = {"model_params": {"default": self.config["model_params"].copy()}}
            self.config["grid_search"] = construct_gs_params(default)
        else:
            self.config["grid_search"] = construct_gs_params(self.config["grid_search"])

        for i, (train_df, test_df) in enumerate(outer_folds):
            self.log.info("Number of train samples : {}".format(len(train_df)))
            self.log.info("Number of test samples : {}".format(len(test_df)))

            self.log.info(
                "Performing {} folds of crossvalidation on test fold {}".format(self.config["inner_kfolds"], i))

            # Create folder for test fold
            test_dir = "{}/test_{}".format(self.run_dir, i)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)
            test_dirs.append(test_dir)
            training_results = []
            for j in range(len(self.config["grid_search"])):
                # Set the config params based on grid search
                for k, v in self.config["grid_search"][j].items():
                    # Ensure not to override own grid search
                    if k != "grid_search":
                        self.config[k] = v
                # Create folder for cv runs
                gs_dir = "{}/cv_{}".format(test_dir, j)
                if not os.path.exists(gs_dir):
                    os.mkdir(gs_dir)
                # Save configuration file for this cv
                with open(f"{gs_dir}/config.txt", "w") as f:
                    f.write("{" + self._sanitise_config(self.config) + "}")
                if self.config["inner_kfolds"] > 1:
                    # Get folds
                    folds = train_df.get_k_splits(self.config["inner_kfolds"], self.config["stratified"],
                                                  self.config["tv_split_seed"])
                    # Evaluate across K folds
                    fold_dirs = []
                    mean_val_metric = []
                    for k, (train_fold, val_fold) in enumerate(folds):
                        # Prepare logging for current fold
                        fold_dir = "{}/fold_{}".format(gs_dir, k)
                        if not os.path.exists(fold_dir):
                            os.mkdir(fold_dir)
                        self.fold_logger = self._get_logger("fold_logger", fold_dir)
                        # Perform fold training
                        val_result = self._fold_run(fold_dir, train_fold, val_fold, test_df)
                        mean_val_metric.append(val_result)
                        fold_dirs.append(fold_dir)
                    # Collate results across folds
                    for fold_collator in self.config["fold_collators"]:
                        fold_collator({"results": fold_dirs, "save_dir": gs_dir})
                    # Save validation metrics to select best model
                    training_results.append(
                        {k: np.mean([mean_val_metric[i][k] for i in range(len(mean_val_metric))]) for k, _ in
                         mean_val_metric[0].items()})
                else:
                    # No K folds so treat it as just train test split with validation split if provided
                    train_fold, val_fold = train_df.get_train_test_split(1 - self.config["validation_prop"],
                                                                         self.config["stratified"],
                                                                         self.config["tv_split_seed"])
                    # Prepare logging for current folder
                    self.fold_logger = self._get_logger("fold_logger", gs_dir)
                    # Perform training
                    val_metric = self._fold_run(gs_dir, train_fold, val_fold, [])
                    training_results.append(val_metric)
            # Compute test metrics on best hyperparameters based on validation metric
            if len(self.config["val_metric"]) > 0:
                training_results = pd.DataFrame(training_results)
                for metric in training_results.columns:
                    best_p = training_results[metric].idxmax()
                    for k, v in self.config["grid_search"][best_p].items():
                        # Ensure not to override own grid search
                        if k != "grid_search":
                            self.config[k] = v
                    best_dir = f"{test_dir}/best_{metric}"
                    gs_dirs.append(best_dir)
                    gs_params.append(self.config["grid_search"][best_p]["model_params_choice"])
                    if not os.path.exists(best_dir):
                        os.mkdir(best_dir)
                    with open(f"{best_dir}/config.txt", "w") as f:
                        f.write("{" + self._sanitise_config(self.config) + "}")
                    if self.config["hold_out_validation_for_final_fit"]:
                        train_fold, val_fold = train_df.get_train_test_split(1 - self.config["validation_prop"],
                                                                             self.config["stratified"],
                                                                             self.config["tv_split_seed"])
                        self.fold_logger = self._get_logger("fold_logger", best_dir)
                        self._fold_run(best_dir, train_fold, val_fold, test_df)
                    else:
                        train_fold, val_fold = train_df.get_train_test_split(1, self.config["stratified"],
                                                                             self.config["tv_split_seed"])
                        self.fold_logger = self._get_logger("fold_logger", best_dir)
                        self._fold_run(best_dir, train_fold, val_fold, test_df)
        for gsc in self.config["grid_search_collators"]:
            gsc({"gs_dirs": gs_dirs, "params": gs_params, "save_dir": self.run_dir, "test_dirs": test_dirs})
        if "external_datasets" in self.config:
            self._run_external_validation()

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

    def _load_data(self, shuffle_seed):
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
        self.data.align_views(self.config["view_alignment_method"], view_aligner, self.config["drop_labels"],
                              shuffle_seed)

    def _summarise_data(self):
        self.log.info("Total number of samples {}".format(len(self.data)))
        for k, v in self.data.data_views.items():
            self.log.info("View {} has {} features".format(k, v.shape[1]))
        self.log.info("{} features aligned across all views".format(len(set(self.data.get_alignment_ids()))))
        self.log.info("{} types of labels".format(self.data.labels.shape[1]))

    def _train(self, train_df, val_df):
        """
        Placeholder function to be overriden by subclasses for different types of model training
        """
        pass

    def _get_modal_hyperparams(self):
        df = pd.read_csv(f"{self.run_dir}/results.csv", index_col=0)

        available_metrics = df["metric"].unique()
        selection_metric = self.config.get("ext_validation_hyperparam_selection_metric")
        if len(available_metrics) == 1:
            # Only one metric recorded, so the choice is unambiguous.
            metric = available_metrics[0]
        elif selection_metric:
            if selection_metric not in available_metrics:
                raise ValueError(
                    f"ext_validation_hyperparam_selection_metric '{selection_metric}' is not "
                    f"present in results.csv; available metrics are {sorted(available_metrics)}."
                )
            metric = selection_metric
        else:
            raise ValueError(
                f"results.csv contains multiple metrics ({sorted(available_metrics)}); set "
                "'ext_validation_hyperparam_selection_metric' to choose which one to use for "
                "external-validation hyperparameter selection."
            )

        df = df[df["metric"] == metric]
        fold_choices = df[["test_fold", "hyperparams"]].drop_duplicates(subset="test_fold")
        modes = fold_choices["hyperparams"].mode()
        if len(modes) == 1:
            choice_key = modes.iloc[0]
            self.log.info(f"External validation: modal hyperparameter choice '{choice_key}' (metric: {metric})")
        else:
            fold_choices = fold_choices.copy()
            fold_choices["fold_num"] = fold_choices["test_fold"].str.rsplit("_", n=1).str[-1].astype(int)
            choice_key = fold_choices.sort_values("fold_num").iloc[0]["hyperparams"]
            self.log.info(f"External validation: tied hyperparameter selection, using fold 0 choice '{choice_key}' (metric: {metric})")
        return next(
            p["model_params"] for p in self.config["grid_search"]
            if p["model_params_choice"] == choice_key
        )

    def _build_and_train_final_model(self, full_train, empty_val, model_params):
        raise NotImplementedError

    def _run_external_validation(self):
        model_params = self.config.get("final_model_params") or self._get_modal_hyperparams()
        self.log.info(f"External validation: model_params = {model_params}")

        full_train, empty_val = self.data.get_train_test_split(1, self.config["stratified"], self.config["tt_split_seed"])
        pre_selection_features = list(full_train.features)
        pre_selection_alignment_ids = list(full_train.alignment_ids)
        feature_selector = copy.deepcopy(self.config["feature_selector"])
        full_train = feature_selector.fit_transform(full_train)
        full_train = self.config["data_augmentor"](full_train)
        preprocessor = copy.deepcopy(self.config["feature_preprocessor"])
        full_train = preprocessor.fit_transform(full_train)

        model = self._build_and_train_final_model(full_train, empty_val, model_params)

        ext_dir = f"{self.run_dir}/external_validation"
        os.makedirs(ext_dir, exist_ok=True)

        for ext_config in self.config["external_datasets"]:
            tag = ext_config["tag"]
            tag_dir = f"{ext_dir}/{tag}"
            os.makedirs(tag_dir, exist_ok=True)

            ext_data = self.config["dataloader"]()
            view_aligner = {}
            for view_name, data_fn, selected_columns, id_col, preprocess_fn, aligner in ext_config["views"]:
                ext_data.load_data_view(view_name, os.path.join(self.config["data_dir"], data_fn),
                                        selected_columns, id_col, preprocess_fn)
                view_aligner[view_name] = aligner
            for label_fn, id_col in ext_config["labels"]:
                ext_data.load_data_label(os.path.join(self.config["data_dir"], label_fn), id_col)
            ext_data.align_views(self.config["view_alignment_method"], view_aligner,
                                 self.config["drop_labels"], drop_zero_label_cols=False,
                                 shuffle_seed=self.config["shuffle_seed"])

            ext_df = pd.DataFrame(ext_data.xs, columns=ext_data.features, index=ext_data.ids)
            ext_df = ext_df.reindex(columns=pre_selection_features, fill_value=0.0)
            ext_data.xs = ext_df.to_numpy(dtype=np.float32)
            ext_data.features = pre_selection_features
            ext_data.alignment_ids = pre_selection_alignment_ids

            ext_data = feature_selector.transform(ext_data)
            ext_data = preprocessor.transform(ext_data)

            preds = model.predict(ext_data.xs).reshape(ext_data.ys.shape)

            label_names = ext_data.get_labels()
            pd.DataFrame(
                np.concatenate([ext_data.ys, preds], axis=1),
                columns=label_names + [f"{l}_pred" for l in label_names],
                index=ext_data.ids,
            ).to_csv(f"{tag_dir}/predictions.csv")

            metrics = self.config.get("external_validation_metrics", {})
            if metrics:
                ext_task = self.config.get("external_validation_task", "individual")
                row = {}
                for metric_name, metric_fn in metrics.items():
                    if ext_task == "individual":
                        for i, label in enumerate(label_names):
                            is_na = np.isnan(ext_data.ys[:, i])
                            y_true, y_pred = ext_data.ys[~is_na, i], preds[~is_na, i]
                            row[f"{label}_{metric_name}"] = metric_fn(y_true, y_pred)
                    elif ext_task == "group":
                        is_na = np.isnan(ext_data.ys).any(axis=1)
                        y_true = ext_data.ys[~is_na]
                        y_pred = preds[~is_na]
                        row[metric_name] = metric_fn(y_true, y_pred)
                pd.DataFrame([row]).to_csv(f"{tag_dir}/metrics.csv", index=False)

            self.log.info(f"External validation '{tag}': {len(ext_data.ids)} samples -> {tag_dir}")

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
        results = {"train_preds": train_preds, "val_preds": val_preds, "test_preds": test_preds,
                   "train_df": train_fold, "val_df": val_fold, "test_df": test_fold, "train_hx": train_hx,
                   "save_dir": fold_dir, "model": model, "feature_preprocessor": preprocessor,
                   "feature_selector": feature_selector}
        # Process results as specified
        self.fold_logger.info("Saving results")
        for result_processor in self.config["results_processors"]:
            result_processor(results)
        self.fold_logger.handlers.clear()
        # return validation metrics
        if len(val_fold) > 0:
            return {k: v(results) for k, v in self.config["val_metric"].items()}


class TFPipeline(Pipeline):
    """
    Trains a TensorFlow model
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.nn_model = None

    def _train(self, train_df, val_df):
        if self.nn_model is None:
            self.nn_model = TFModel(
                self.config["run_id"],
                self.config["model"],
                self.config["model_params"],
                self.config["fitting_params"]
            )
        else:
            self.nn_model.set_params(
                self.config["run_id"],
                self.config["model"],
                self.config["model_params"],
                self.config["fitting_params"]
            )
        model, train_hx = self.nn_model.fit(train_df, val_df, self.config["rng_seed"])
        return model, train_hx

    def _build_and_train_final_model(self, full_train, empty_val, model_params):
        tf_model = TFModel(
            self.config["run_id"],
            self.config["model"],
            model_params,
            self.config["fitting_params"],
        )
        model, _ = tf_model.fit(full_train, empty_val, self.config["rng_seed"])
        return model


class MLPipeline(Pipeline):
    def __init__(self, config: dict):
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
        model = SKModelWrapper(self.config["model"], self.config["task"], self.config["model_params"])
        model.fit(train_df.xs, train_df.ys)
        return model, None

    def _build_and_train_final_model(self, full_train, empty_val, model_params):
        np.random.seed(self.config["rng_seed"])
        model = SKModelWrapper(self.config["model"], self.config["task"], model_params)
        model.fit(full_train.xs, full_train.ys)
        return model


class SKModelWrapper:
    def __init__(self, model, task, params):
        self.model = model(**params)
        self.task = task

    def fit(self, xs, ys):
        self.model.fit(xs, ys) if ys.shape[1] > 1 else self.model.fit(xs, ys.ravel())

    def predict(self, xs):
        if self.task == "binary classification":
            results = self.model.predict_proba(xs)
            return results[:, 1]
        else:
            results = self.model.predict(xs)
            return results


def construct_gs_params(params):
    # Already a list of expanded param dicts — pass through
    if isinstance(params, list):
        return params

    params = params.copy()
    cur_param = list(params.keys())[0]
    cur_vals = params.pop(cur_param)
    if len(params) == 0:
        out = [{cur_param: v, f"{cur_param}_choice": k} for k, v in cur_vals.items()]
        return out
    else:
        out = construct_gs_params(params)
        return [gs_param | {cur_param: v, f"{cur_param}_choice": k} for k, v in cur_vals.items() for gs_param in out]


class IdentityProcessor:
    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

    def fit(self, dataset):
        pass

    def transform(self, dataset):
        return dataset
