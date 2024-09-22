import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from piml.models import GAMINetRegressor, GAMINetClassifier
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from pygam import terms, s, f
from pygam.pygam import LogisticGAM, LinearGAM
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from baseline.lemeln_nam.nam.wrapper import NAMClassifier, NAMRegressor
from igann import IGANN
import torch
import os

# ExNN does not provide support for ARM architecture.
# Thus please run code that includes runs with ExNN on a x86_64 machine
if not os.uname().machine == "arm64":
    from baseline.exnn.exnn import ExNN


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


class Model:

    def __init__(
        self,
        model_name,
        task,
        arg_dict,
        num_cols=None,
        cat_cols=None,
    ):
        self.model_name = model_name
        self.task = task
        self.arg_dict = arg_dict
        self.num_cols = num_cols
        self.cat_cols = cat_cols  # TabNet needs to know on which i a categorical column is and which dimension it has.
        self.model = self._get_model()
        # some models like exnn and GAMINET need to know which columns are categorical and which are numerical for a
        # reason internally. Other models do not need to know.

    def fit(self, X_train, y_train):
        if not self.model_name == "MLP" and not isinstance(X_train, pd.DataFrame):
            raise ValueError(
                "X_train must be a pandas DataFrame to identify categorical columns"
            )

        if self.model_name == "EXNN":
            self.model = self._load_exnn(X_train, y_train)
            self.model.fit(self.X_train, self.y_train)
            y_proba = self.model.predict(self.X_train)
            if self.task == "classification":
                self.best_threshold = self._optimize_threshold(y_proba)

        elif self.model_name == "PYGAM":
            tms = terms.TermList(
                *[
                    (
                        f(i)
                        if X_train.columns[i] in self.cat_cols
                        else s(
                            i,
                            n_splines=self.arg_dict["n_splines"],
                            lam=self.arg_dict["lam"],
                        )
                    )
                    for i in range(X_train.shape[1])
                ]
            )
            if self.task == "classification":
                self.model = LogisticGAM(
                    tms
                )  # tol=1e-4 is the default, but the internal svd fails to converge on some datasets
            elif self.task == "regression":
                self.model = LinearGAM(
                    tms
                )  # tol=1e-4 is the default, but the internal svd fails to converge on some datasets
            self.model.fit(X_train, y_train)
        elif self.model_name == "TABNET":
            if self.task == "regression":
                X_train, X_eval, y_train, y_eval = train_test_split(
                    X_train,
                    y_train,
                    test_size=0.10,
                    random_state=1337,
                )
                self.model.fit(
                    X_train.values,
                    y_train.values.reshape(-1, 1),
                    eval_set=[(X_eval.values, y_eval.values.reshape(-1, 1))],
                    eval_name=["eval"],
                    eval_metric=["rmse"],
                    patience=20,
                    batch_size=int((1 / 10) * X_train.shape[0]),
                )  # TABNET requires numpy arrays instead of pd.DataFrame
            elif self.task == "classification":
                X_train, X_eval, y_train, y_eval = train_test_split(
                    X_train,
                    y_train,
                    test_size=0.10,
                    stratify=y_train,
                    random_state=1337,
                )
                self.model.fit(
                    X_train.values,
                    y_train.values,
                    eval_set=[(X_eval.values, y_eval.values)],
                    eval_name=["eval"],
                    eval_metric=["auc"],
                    patience=20,
                    batch_size=int((1 / 10) * X_train.shape[0]),
                )  # TABNET requires numpy arrays instead of pd.DataFrame
        elif self.model_name == "CATBOOST":
            if self.task == "classification":
                self.model = CatBoostClassifier(
                    random_seed=0,
                    # task_type="GPU" if device == "cuda" else None,
                    n_estimators=self.arg_dict["n_estimators"],
                    eta=self.arg_dict["eta"],
                    max_depth=self.arg_dict["max_depth"],
                    # internal enc is slower
                    # cat_features=self.cat_cols,
                    verbose=0,
                )
            elif self.task == "regression":
                self.model = CatBoostRegressor(
                    random_seed=0,
                    # task_type="GPU" if device == "cuda" else None,
                    n_estimators=self.arg_dict["n_estimators"],
                    eta=self.arg_dict["eta"],
                    max_depth=self.arg_dict["max_depth"],
                    # internal enc is slower
                    # cat_features=self.cat_cols,
                    verbose=0,
                )
            # the categorical data should be in pandas as a type categorical instead of object, however there is no speedup using this
            # X_train[self.cat_cols] = X_train[self.cat_cols].astype("category")
            X_train[self.num_cols] = X_train[self.num_cols].astype("float")

            train_data = Pool(X_train, y_train)
            self.model.fit(train_data)
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if self.model_name == "TABNET":
            return self.model.predict(
                X_test.values
            )  # TABNET requires numpy arrays instead of pd.DataFrame
        elif self.model_name == "CATBOOST":
            test_data = Pool(X_test)
            return self.model.predict(test_data)
        elif self.model_name == "EXNN":
            # EXNN returns only logits in one column. We need to convert it to target values
            # take self.best_threshold and apply it to the logits
            y_pred = self.model.predict(X_test)
            if self.task == "classification":
                y_pred = np.where(y_pred > self.best_threshold, 1, 0)
        else:
            return self.model.predict(X_test)

    def predict_proba(self, X_test):
        if self.model_name == "EXNN":
            return self.model.predict(X_test)
        elif self.model_name == "IGANN":
            return self.model.predict_proba(X_test)[:, 1]
        elif self.model_name == "TABNET":
            return self.model.predict_proba(X_test.values)
        else:
            return self.model.predict_proba(X_test)

    def _get_model(self):

        if "LR" in self.model_name:
            if self.task == "classification":
                if self.arg_dict["penalty"] == "elasticnet":
                    return LogisticRegression(
                        C=self.arg_dict["C"],
                        penalty=self.arg_dict["penalty"],
                        class_weight=self.arg_dict["class_weight"],
                        solver=self.arg_dict["solver"],
                        l1_ratio=self.arg_dict["l1_ratio"],
                        max_iter=self.arg_dict["max_iter"],
                        n_jobs=-1,
                        random_state=0,
                    )
                else:
                    return LogisticRegression(
                        C=self.arg_dict["C"],
                        penalty=self.arg_dict["penalty"],
                        class_weight=self.arg_dict["class_weight"],
                        solver=self.arg_dict["solver"],
                        max_iter=self.arg_dict["max_iter"],
                        n_jobs=-1,
                        random_state=0,
                    )
        elif "ELASTICNET" in self.model_name:
            if self.task == "regression":
                # ridge regression would be the default, so we need to set l1_ratio to 0 in case of default
                # we utilize elasticnet with l1 ratio of [0, 1] to fit lasso, ridge and everything between
                return ElasticNet(
                    alpha=self.arg_dict["alpha"],
                    l1_ratio=self.arg_dict["l1_ratio"],
                    max_iter=2000,
                    random_state=0,
                )
        elif "RF" in self.model_name:
            if self.task == "classification":
                return RandomForestClassifier(
                    n_estimators=self.arg_dict["n_estimators"],
                    max_depth=self.arg_dict["max_depth"],
                    class_weight=self.arg_dict["class_weight"],
                    n_jobs=-1,
                    random_state=0,
                )
            elif self.task == "regression":
                return RandomForestRegressor(
                    n_estimators=self.arg_dict["n_estimators"],
                    max_depth=self.arg_dict["max_depth"],
                    n_jobs=-1,
                    random_state=0,
                )

        elif "DT" in self.model_name:
            if self.task == "classification":
                return DecisionTreeClassifier(
                    max_depth=self.arg_dict["max_depth"],
                    max_leaf_nodes=self.arg_dict["max_leaf_nodes"],
                    class_weight=self.arg_dict["class_weight"],
                    splitter=self.arg_dict["splitter"],
                    random_state=0,
                )

            elif self.task == "regression":
                return DecisionTreeRegressor(
                    max_depth=self.arg_dict["max_depth"],
                    max_leaf_nodes=self.arg_dict["max_leaf_nodes"],
                    splitter=self.arg_dict["splitter"],
                    random_state=0,
                )

        elif "MLP" in self.model_name:
            if self.task == "classification":
                return MLPClassifier(
                    hidden_layer_sizes=self.arg_dict["hidden_layer_sizes"],
                    alpha=self.arg_dict["alpha"],
                    activation=self.arg_dict["activation"],
                    n_iter_no_change=100,
                    learning_rate="constant",
                    solver="adam",
                    max_iter=600,
                    early_stopping=True,
                    random_state=0,
                )
            elif self.task == "regression":
                return MLPRegressor(
                    hidden_layer_sizes=self.arg_dict["hidden_layer_sizes"],
                    alpha=self.arg_dict["alpha"],
                    activation=self.arg_dict["activation"],
                    n_iter_no_change=100,
                    learning_rate="constant",
                    solver="adam",
                    max_iter=600,
                    early_stopping=True,
                    random_state=0,
                )
        elif "XGB" in self.model_name:
            if self.task == "classification":
                return XGBClassifier(
                    n_estimators=self.arg_dict["n_estimators"],
                    max_depth=self.arg_dict["max_depth"],
                    learning_rate=self.arg_dict["learning_rate"],
                    random_state=0,
                )
            elif self.task == "regression":
                return XGBRegressor(
                    n_estimators=self.arg_dict["n_estimators"],
                    max_depth=self.arg_dict["max_depth"],
                    learning_rate=self.arg_dict["learning_rate"],
                    random_state=0,
                )

        elif "PYGAM" in self.model_name:
            if self.task == "classification":
                # there is a dedicated function for this, because pygam is special and needs information about the data first
                return None
            elif self.task == "regression":
                return None
        elif "IGANN" in self.model_name:
            if self.task == "classification":
                return IGANN(
                    interactions=self.arg_dict["interactions"],
                    elm_scale=self.arg_dict["elm_scale"],
                    boost_rate=self.arg_dict["boost_rate"],
                    n_hid=10,
                    elm_alpha=1,
                    elm_scale_inter=0.5,
                    verbose=1,
                    device=device,
                    optimize_threshold=False,
                    random_state=1,
                )
            elif self.task == "regression":
                return IGANN(
                    task="regression",
                    interactions=self.arg_dict["interactions"],
                    elm_scale=self.arg_dict["elm_scale"],
                    boost_rate=self.arg_dict["boost_rate"],
                    n_hid=10,
                    elm_alpha=1,
                    elm_scale_inter=0.5,
                    verbose=1,
                    device=device,
                    random_state=1,
                )
        elif "EBM" in self.model_name:
            if self.task == "classification":
                return ExplainableBoostingClassifier(
                    max_bins=self.arg_dict["max_bins"],
                    interactions=self.arg_dict["interactions"],
                    outer_bags=self.arg_dict["outer_bags"],
                    inner_bags=self.arg_dict["inner_bags"],
                    random_state=42,
                )
            elif self.task == "regression":
                return ExplainableBoostingRegressor(
                    max_bins=self.arg_dict["max_bins"],
                    interactions=self.arg_dict["interactions"],
                    outer_bags=self.arg_dict["outer_bags"],
                    inner_bags=self.arg_dict["inner_bags"],
                    random_state=42,
                )
        elif "GAMINET" in self.model_name:
            if self.task == "classification":
                return GAMINetClassifier(
                    batch_size=1024,
                    interact_num=self.arg_dict["interact_num"],
                    activation_func=self.arg_dict["activation_func"],
                    reg_clarity=self.arg_dict["reg_clarity"],
                    warm_start=False,
                    max_epochs=(3000, 1000, 1000),
                    verbose=True,
                    device="cpu",  # cuda takes est. 2-3x longer
                    random_state=0,
                )
            elif self.task == "regression":
                return GAMINetRegressor(
                    batch_size=1024,
                    interact_num=self.arg_dict["interact_num"],
                    activation_func=self.arg_dict["activation_func"],
                    reg_clarity=self.arg_dict["reg_clarity"],
                    warm_start=False,
                    max_epochs=(3000, 1000, 1000),
                    verbose=True,
                    device="cpu",  # cuda takes est. 3x longer
                    random_state=0,
                )
        elif "TABNET" in self.model_name:
            if self.task == "classification":
                return TabNetClassifier(
                    seed=0,
                    n_a=self.arg_dict["n_a_and_d"],
                    n_d=self.arg_dict["n_a_and_d"],
                    n_steps=self.arg_dict["n_steps"],
                    gamma=self.arg_dict["gamma"],
                    # cat_idxs=cat_idxs,
                    # cat_dims=cat_dims,
                    # cat_emb_dim=1,
                    # scheduler_fn=torch.optim.lr_scheduler.StepLR,
                    # scheduler_params={"step_size": 10, "gamma": 0.95},
                    device_name=device,
                )
            elif self.task == "regression":
                return TabNetRegressor(
                    seed=0,
                    n_a=self.arg_dict["n_a_and_d"],
                    n_d=self.arg_dict["n_a_and_d"],
                    n_steps=self.arg_dict["n_steps"],
                    gamma=self.arg_dict["gamma"],
                    # cat_idxs=cat_idxs,
                    # cat_dims=cat_dims,
                    # cat_emb_dim=1,
                    # scheduler_fn=torch.optim.lr_scheduler.StepLR,
                    # scheduler_params={"step_size": 10, "gamma": 0.95},
                    device_name=device,
                )

        elif "CATBOOST" in self.model_name:
            return None
        elif "EXNN" in self.model_name:
            # there is a dedicated function for this, because exnn is special and needs information about the data first
            return None
        elif "NAM" in self.model_name:
            if self.task == "classification":
                return NAMClassifier(
                    # vary
                    num_learners=self.arg_dict["num_learners"],
                    num_basis_functions=self.arg_dict["num_basis_functions"],
                    dropout=self.arg_dict["dropout"],
                    lr=self.arg_dict["lr"],
                    # fixed
                    # metric='auroc',
                    # early_stop_mode='max',
                    monitor_loss=True,
                    n_jobs=8,
                    device=device,
                    batch_size=4096,
                    random_state=42,
                )
            elif self.task == "regression":
                return NAMRegressor(
                    # vary
                    num_learners=self.arg_dict["num_learners"],
                    num_basis_functions=self.arg_dict["num_basis_functions"],
                    dropout=self.arg_dict["dropout"],
                    lr=self.arg_dict["lr"],
                    # fixed
                    metric="rmse",
                    early_stop_mode="min",
                    monitor_loss=False,
                    n_jobs=8,
                    device=device,
                    batch_size=4096,
                    random_state=42,
                )
        else:
            raise ValueError("Model not supported")

    def _load_exnn(self, X_train, y_train):
        meta_info = {
            f"{col}": {"type": "continuous"} for i, col in enumerate(self.num_cols)
        }
        # extend this by the categorical columns
        meta_info.update(
            {f"{col}": {"type": "continuous"} for i, col in enumerate(self.cat_cols)}
        )
        meta_info.update({"Y": {"type": "target"}})

        for i, (key, item) in enumerate(meta_info.items()):
            if item["type"] == "target":
                # save the y_train for use in fitting the exnn
                self.y_train = np.array(y_train).reshape(-1, 1)
                # y_test = np.array(y_test).reshape(-1, 1)
            else:
                sx = MinMaxScaler((0, 1))
                sx.fit([[0], [1]])
                # save the X_train for use in fitting the exnn
                self.X_train = np.array(X_train)
                # X_test = np.array(X_test)
                self.X_train[:, [i]] = sx.transform(np.array(X_train)[:, [i]])
                # X_test[:, [i]] = sx.transform(np.array(X_test)[:, [i]])
                meta_info[key]["scaler"] = sx

        if self.task == "classification":
            return ExNN(
                meta_info=meta_info,
                subnet_num=self.arg_dict["subnet_num"],
                l1_proj=self.arg_dict["l1_proj"],
                l1_subnet=self.arg_dict["l1_subnet"],
                task_type="Classification",
                batch_size=512,
                verbose=True,
                random_state=0,
            )

        elif self.task == "regression":
            return ExNN(
                meta_info=meta_info,
                subnet_num=self.arg_dict["subnet_num"],
                l1_proj=self.arg_dict["l1_proj"],
                l1_subnet=self.arg_dict["l1_subnet"],
                task_type="Regression",
                batch_size=512,
                verbose=True,
                random_state=0,
            )

    def _optimize_threshold(self, y_proba):
        fpr, tpr, trs = roc_curve(self.y_train, y_proba)

        roc_scores = []
        thresholds = []
        for thres in trs:
            thresholds.append(thres)
            y_pred = np.where(y_proba > thres, 1, 0)
            # Apply desired utility function to y_preds, for example accuracy.
            roc_scores.append(roc_auc_score(self.y_train.squeeze(), y_pred.squeeze()))
        # convert roc_scores to numpy array
        roc_scores = np.array(roc_scores)
        # get the index of the best threshold
        ix = np.argmax(roc_scores)
        # get the best threshold
        return thresholds[ix]
