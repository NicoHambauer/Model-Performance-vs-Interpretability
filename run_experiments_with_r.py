# SPDX-FileCopyrightText: 2024 Nico Hambauer, Sven Kruschel
#
# SPDX-License-Identifier: MIT

import os
import re
import warnings

import numpy as np
import pandas as pd
import json
import itertools

from numpy.linalg import LinAlgError
from pygam.utils import OptimizationError
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from datetime import datetime

from logging_reports import JournalLogger
from load_datasets import Dataset
from model_with_r import Model
from rpy2 import rinterface_lib

"""
    Traditional Models:
    - LR (Elasticnet, Lasso, Ridge)
    - RF
    - XGB
    - DT
    GAMs:
    - PYGAM
    - EBM
    - NAM
    - GAMINET
    - EXNN
    - IGANN
    - MGCV (R)
    - GAMens
"""

random_state = 42
verbose = 2
n_folds = 5

hyperparameter_config_file = "./default_hyperparams.json"
with open(hyperparameter_config_file, "r") as read_file:
    hpo_grid = json.load(read_file)

tasks = ["classification", "regression"]  #

traditional_models_to_run = [
    "TABNET",
    "LR",
    "DT",
    "RF",
    "XGB",
    "MLP",
    "CATBOOST",
]
gam_models_to_run = [
    "PYGAM",
    "EBM",
    "NAM",
    "GAMINET",
    "EXNN",
    "IGANN",
    "MGCV_SPLINE" "GAMENS",
]

classification_datasets = [
    "college",
    "water",
    "stroke",
    "telco",
    "compas",
    "fico",
    "adult",
    "bank",
    "airline",
    "weather",
]
regression_datasets = [
    "car",
    "student",
    "productivity",
    "medical",
    "crimes",
    "crab",
    "wine",
    "bike",
    "housing",
    "diamond",
]

for task in tasks:
    directory = f"./GAMens_experiments_500_iter/default/{task}"
    # read or create the best hpo config csvs
    best_hpo_config_csvs = []
    for i in range(1, n_folds + 1):
        if not os.path.exists(f"{directory}/hpo_best_config_Fold_{i}.csv"):
            best_hpo_config_csvs.append(
                pd.DataFrame(
                    index=classification_datasets + regression_datasets,
                    columns=traditional_models_to_run + gam_models_to_run,
                )
            )
        else:
            best_hpo_config_csvs.append(
                pd.read_csv(
                    f"{directory}/hpo_best_config_Fold_{i}.csv", index_col=0, header=0
                )
            )

    datasets_to_run_on = None
    if task == "classification":
        datasets_to_run_on = classification_datasets
    elif task == "regression":
        datasets_to_run_on = regression_datasets

    for model_name in traditional_models_to_run + gam_models_to_run:
        for dataset_name in datasets_to_run_on:
            if model_name == "LR" and task == "regression":
                # use the sklearn Elasticnet for regression instead of LR
                model_name = "ELASTICNET"

            keys, values = zip(*hpo_grid[model_name].items())
            # Compute all combinations from the hpo grid
            permutations_dicts = [
                dict(zip(keys, v)) for v in itertools.product(*values)
            ]

            if model_name == "LR":
                condition = lambda args: (
                    (args["solver"] == "lbfgs" and args["penalty"] == "l1")
                    or (args["solver"] == "lbfgs" and args["penalty"] == "elasticnet")
                    or
                    # ‘lbfgs’ only works with [‘l2’, None]
                    (args["solver"] == "liblinear" and args["penalty"] == "elasticnet")
                    or (args["solver"] == "liblinear" and args["penalty"] == "none")
                    or
                    # ‘liblinear’ only works with  [‘l1’, ‘l2’]
                    (
                        isinstance(args["l1_ratio"], float)
                        and args["penalty"] != "elasticnet"
                    )
                    or
                    # l1_ratio is only used when penalty is elasticnet
                    (args["l1_ratio"] is None and args["penalty"] == "elasticnet")
                    # when elasticnet is used l1_ratio must be not None
                )

                permutations_dicts = [
                    item for item in permutations_dicts if not condition(item)
                ]

            if (model_name == "RF" or model_name == "DT") and task == "regression":
                condition = lambda args: args["class_weight"] == "balanced"
                permutations_dicts = [
                    item for item in permutations_dicts if not condition(item)
                ]

            logger = JournalLogger()
            logger.set_global_result_dir(directory)

            print("\n", "#" * 3, f"Run experiment on {dataset_name}", "#" * 3)

            # load dataset
            dataset = Dataset(dataset_name, model_name)

            X = dataset.X
            y = dataset.y

            # We use Inner Split - outer Cross validation
            # The purpose is in the outer cv to get an estimation of the test error.
            # The inner split val is used to tune the hyperparameters of the model.
            # we made the tradeoff of using an inner split instead another cv loop to reduce the computational cost.
            outer_cv = None
            if task == "classification":
                outer_cv = StratifiedKFold(
                    n_splits=n_folds, shuffle=True, random_state=random_state
                )
            elif task == "regression":
                outer_cv = KFold(
                    n_splits=n_folds, shuffle=True, random_state=random_state
                )

            for fold_i, (train_val_idx, test_idx) in enumerate(outer_cv.split(X, y)):

                print(
                    "\n",
                    "-" * 5,
                    "Model:",
                    model_name,
                    "-- Fold:",
                    fold_i + 1,
                    "/",
                    n_folds,
                    "-" * 5,
                )
                X_train_val, y_train_val = X.iloc[train_val_idx], y.iloc[train_val_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

                if task == "regression":
                    y_scaler = StandardScaler()
                    # scale the target out of sample for regression
                    y_train_val = pd.Series(
                        y_scaler.fit_transform(
                            y_train_val.values.reshape(-1, 1)
                        ).flatten()
                    )
                    y_test = pd.Series(
                        y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
                    )

                # one hot encoder pipeline drops the original categorical columns if binary. That means two
                # categories male and female become one column e.g. female = 0 or 1
                cat_step = (
                    "ohe",
                    OneHotEncoder(
                        sparse=False, handle_unknown="ignore", drop="if_binary"
                    ),
                )

                if model_name == "CATBOOST":
                    cat_step = ("identity", "passthrough")

                cat_pipe = Pipeline([cat_step])
                num_pipe = Pipeline([("scaler", StandardScaler())])
                transformers = [
                    ("cat", cat_pipe, dataset.categorical_cols),
                    ("num", num_pipe, dataset.numerical_cols),
                ]
                ct = ColumnTransformer(transformers=transformers)

                # split val the data into train and val
                if task == "classification":
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train_val,
                        y_train_val,
                        test_size=0.25,
                        stratify=y_train_val,
                        random_state=1337,
                    )
                elif task == "regression":
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train_val, y_train_val, test_size=0.25, random_state=1337
                    )

                ct.fit(X_train)

                X_train = pd.DataFrame(
                    ct.transform(X_train), columns=ct.get_feature_names_out()
                )
                X_val = pd.DataFrame(
                    ct.transform(X_val), columns=ct.get_feature_names_out()
                )

                print("Dataset Categorical Columns:", dataset.categorical_cols)
                print("Dataset Numerical Columns:", dataset.numerical_cols)

                transformed_feature_names = ct.get_feature_names_out()

                # Maps to store the original to new feature names
                mapped_categorical_cols = []
                mapped_numerical_cols = []

                for original_col in dataset.categorical_cols:
                    # Find all transformed categorical features for this original column
                    mapped_features = [
                        feature
                        for feature in transformed_feature_names
                        if feature.startswith(f"cat__{original_col}")
                    ]
                    mapped_categorical_cols.extend(mapped_features)

                for original_col in dataset.numerical_cols:
                    # Assuming numerical features keep their original column name in the prefix
                    mapped_feature = next(
                        (
                            feature
                            for feature in transformed_feature_names
                            if feature.endswith(f"__{original_col}")
                        ),
                        None,
                    )
                    if mapped_feature:
                        mapped_numerical_cols.append(mapped_feature)

                # Now you have the correctly mapped and ordered lists of transformed feature names
                print("Mapped Categorical Columns:", mapped_categorical_cols)
                print("Mapped Numerical Columns:", mapped_numerical_cols)

                if model_name == "MLP":
                    X_train = X_train.values
                    X_val = X_val.values

                if verbose == 1:
                    print("")

                best_hp_config = None
                best_loss = np.inf
                training_time_of_best_model = np.inf
                timings_hpo = []
                # tuning hyperparameters in case of multiple hyperparameter candidates
                logger.set_current_dataset_model_dir(dataset_name, model_name)

                for id, arg_dict in enumerate(permutations_dicts):
                    # print the progress with replacing in line all the time
                    if verbose == 1:
                        print(
                            "\r",
                            "Progress: ",
                            id + 1,
                            "/",
                            len(permutations_dicts),
                            end="",
                        )
                    elif verbose == 2:
                        print("-" * 20)
                        print(arg_dict)

                    # define the model
                    model = Model(
                        model_name,
                        task,
                        arg_dict,
                        num_cols=mapped_numerical_cols,
                        cat_cols=mapped_categorical_cols,
                    )

                    start_training_time = datetime.now()

                    try:
                        # fit the model
                        model.fit(X_train, y_train)
                    except (
                        LinAlgError,
                        OptimizationError,
                        rinterface_lib.embedded.RRuntimeError,
                    ) as e:
                        print(e)
                        if isinstance(e, rinterface_lib.embedded.RRuntimeError):
                            # Define the log message
                            if model_name == "MGCV_SPLINE":
                                log_message = f"{task} -- {model_name} -- {dataset_name} -- Fold: {fold_i} -- Spline: {model.model.spline_type} : {e}"
                            else:
                                log_message = f"{task} -- {model_name} -- {dataset_name} -- Fold: {fold_i}: {e}"

                            # Open the log file in append mode
                            with open("log_file.txt", "a") as file:
                                # Write the log message to the file with a newline at the end
                                file.write("Fit-Error: " + log_message + "\n")
                        elif isinstance(e, (LinAlgError, OptimizationError)):
                            warnings.warn(
                                "Training with this hp combination, Error in Gaminet (Optimization Error, warm start) or Pygam (LinAlgError) possible"
                            )

                        continue

                    training_time = (
                        datetime.now() - start_training_time
                    ).total_seconds()
                    timings_hpo.append(training_time)

                    if task == "regression":
                        # calculate the mse
                        y_pred = model.predict(X_val)
                        mse = mean_squared_error(y_val, y_pred)
                        if mse < best_loss:
                            best_hp_config = arg_dict
                            best_loss = mse
                            training_time_of_best_model = training_time
                        print("mse: ", mse, "best_loss: ", best_loss)
                    elif task == "classification":
                        # calculate the loss
                        y_pred = model.predict(X_val)
                        ce_loss = log_loss(y_val, y_pred)
                        if ce_loss < best_loss:
                            best_hp_config = arg_dict
                            best_loss = ce_loss
                            training_time_of_best_model = training_time
                        print("ce_loss: ", ce_loss, "best_loss: ", best_loss)

                best_hpo_string = (
                    str(best_hp_config)
                    .replace("{", "")
                    .replace("}", "")
                    .replace(",", "\n")
                )
                best_hpo_config_csvs[fold_i].loc[
                    dataset_name, model_name
                ] = best_hpo_string

                # now take the best hpo config and retrain on X_train_val and y_train_val
                ct_test = ColumnTransformer(transformers=transformers)
                ct_test.fit(X_train_val)
                X_train_val = pd.DataFrame(
                    ct_test.transform(X_train_val),
                    columns=ct_test.get_feature_names_out(),
                )
                X_test = pd.DataFrame(
                    ct_test.transform(X_test), columns=ct_test.get_feature_names_out()
                )

                if model_name == "MLP":
                    X_train_val = X_train_val.values
                    X_test = X_test.values

                transformed_feature_names = ct.get_feature_names_out()

                # Maps to store the original to new feature names
                mapped_categorical_cols = []
                mapped_numerical_cols = []

                for original_col in dataset.categorical_cols:
                    # Find all transformed categorical features for this original column
                    mapped_features = [
                        feature
                        for feature in transformed_feature_names
                        if feature.startswith(f"cat__{original_col}")
                    ]
                    mapped_categorical_cols.extend(mapped_features)

                for original_col in dataset.numerical_cols:
                    # Assuming numerical features keep their original column name in the prefix
                    mapped_feature = next(
                        (
                            feature
                            for feature in transformed_feature_names
                            if feature.endswith(f"__{original_col}")
                        ),
                        None,
                    )
                    if mapped_feature:
                        mapped_numerical_cols.append(mapped_feature)

                # Now you have the correctly mapped and ordered lists of transformed feature names
                print("Mapped Categorical Columns:", mapped_categorical_cols)
                print("Mapped Numerical Columns:", mapped_numerical_cols)

                print("BEST CONFIG: ", best_hp_config)
                try:
                    best_model = Model(
                        model_name,
                        task,
                        best_hp_config,
                        num_cols=mapped_numerical_cols,
                        cat_cols=mapped_categorical_cols,
                    )
                except Exception as e:
                    print(e)
                    if model_name == "MGCV_SPLINE":
                        log_message = f"{task} -- {model_name} -- {dataset_name} -- Fold: {fold_i} -- Spline: {model.model.spline_type} : {e}"
                    else:
                        log_message = f"{task} -- {model_name} -- {dataset_name} -- Fold: {fold_i}: {e}"
                    # Open the log file in append mode
                    with open("log_file.txt", "a") as file:
                        # Write the log message to the file with a newline at the end
                        file.write("Best-Model-Error: " + log_message + "\n")
                    continue
                try:
                    best_model.fit(X_train_val, y_train_val)
                except (
                    OptimizationError,
                    LinAlgError,
                    rinterface_lib.embedded.RRuntimeError,
                ) as e:
                    print(e)
                    if isinstance(e, rinterface_lib.embedded.RRuntimeError):
                        # Define the log message
                        if model_name == "MGCV_SPLINE":
                            log_message = f"{task} -- {model_name} -- {dataset_name} -- Fold: {fold_i} -- Spline: {model.model.spline_type} : {e}"
                        else:
                            log_message = f"{task} -- {model_name} -- {dataset_name} -- Fold: {fold_i}: {e}"
                        # Open the log file in append mode
                        with open("log_file.txt", "a") as file:
                            # Write the log message to the file with a newline at the end
                            file.write("Fit-Best-Model-Error: " + log_message + "\n")
                    warnings.warn(
                        "Training with this hp combination, Error in Gaminet (Optimization Error, warm start) or Pygam (LinAlgError) possible"
                    )
                    continue
                # except TypeError as e:
                #     print(e)
                #     continue
                else:
                    # evaluate the retrained best model on the hold out dataset
                    y_pred = best_model.predict(X_test)
                    if task == "classification":
                        y_pred_proba = best_model.predict_proba(X_test)

                if task == "classification":
                    logger.log_classification_report(
                        y_true=y_test, y_pred=y_pred, dataset=dataset, k_fold=fold_i
                    )
                    logger.log_roc_auc(
                        y_true=y_test, y_pred_confidence=y_pred_proba, k_fold=fold_i
                    )
                elif task == "regression":
                    logger.log_regression_report(
                        y_true=y_test, y_pred=y_pred, k_fold=fold_i
                    )

                logger.log_timing(
                    training_time_of_best_model, np.mean(timings_hpo), fold_i
                )

            for i in range(n_folds):
                best_hpo_config_csvs[i].to_csv(
                    f"{directory}/hpo_best_config_Fold_{i + 1}.csv",
                    index=True,
                    header=True,
                )
