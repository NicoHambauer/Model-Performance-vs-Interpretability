import os

# only for local use
# os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.0"
# os.environ["PATH"] = r"C:\Program Files\R\R-4.4.0\bin\x64" + ";" + os.environ["PATH"]

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, ElasticNet
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from interpret import show
from pygam import LogisticGAM, LinearGAM
from pygam import terms, s, f
from piml.models import GAMINetRegressor, GAMINetClassifier

# from gaminet import GAMINet
from baseline.exnn.exnn.exnn import ExNN
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
)
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from load_datasets import Dataset
import tensorflow as tf
import torch
from scipy.stats.mstats import winsorize

import pytorch_lightning as pl

# from baseline.nam.config import defaults
# from baseline.nam.data import NAMDataset
# from baseline.nam.models import NAM, get_num_units
# from baseline.nam.trainer import LitNAM
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from EBMLocalExplainer import EBMLocalExplainer

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from igann import IGANN

# Below the R stuff
from rpy2.robjects import pandas2ri, packages
from arch import MyRSplineClassifier, MyRSplineRegressor


def transform_dataset(
    dataset: Dataset, random_state: int = 1, split=True, model_name=""
):
    X = dataset.X
    y = dataset.y

    if split:
        X, _, y, _ = train_test_split(X, y, test_size=0.33, random_state=random_state)
    else:
        X, y = shuffle(X, y, random_state=1)

    # sparse=False  bei OneHotEncoder
    transformers = [
        (
            "ohe",
            OneHotEncoder(sparse=False, handle_unknown="ignore", drop="if_binary"),
            dataset.categorical_cols,
        ),
        ("num", FunctionTransformer(), dataset.numerical_cols),
    ]

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    ct.fit(X)
    X_original = X
    X = ct.transform(X)

    cat_cols = (
        ct.named_transformers_["ohe"].get_feature_names_out(dataset.categorical_cols)
        if len(dataset.categorical_cols) > 0
        else []
    )
    X = pd.DataFrame(X, columns=np.concatenate((cat_cols, dataset.numerical_cols)))

    scaler_dict = {}
    scaler_dict_gaminet = {}
    X_gaminet = X.copy()
    for c in dataset.numerical_cols:
        # sx = MinMaxScaler((0, 1))
        # sx.fit([[0], [1]])
        # X[c] = sx.transform(X[c].values.reshape(-1, 1))

        mms = MinMaxScaler()
        X_gaminet[c] = mms.fit_transform(X[c].values.reshape(-1, 1))
        scaler_dict_gaminet[c] = mms

        sx = StandardScaler()
        X[c] = sx.fit_transform(X[c].values.reshape(-1, 1))
        # scaler = RobustScaler()
        # X[c] = scaler.fit_transform(X[c].values.reshape(-1, 1))
        scaler_dict[c] = sx

    if task == "regression":
        y_scaler = StandardScaler()
        y = pd.Series(y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten())
    else:
        y_scaler = None

    if model_name == "Gaminet":
        return X, y, X_original, y_scaler, scaler_dict_gaminet
    else:
        return X, y, X_original, y_scaler, scaler_dict


@dataclass
class PlotData:
    _plot_dict: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def add_datapoint(
        self,
        column: str,
        label: str,
        not_given_category_value: float,
        given_category_value: float,
    ):
        if column not in self._plot_dict:
            self._add_column(column)
        self._plot_dict[column][label] = {
            0: not_given_category_value,
            1: given_category_value,
        }

    def _add_column(self, name: str):
        self._plot_dict[name] = {}

    @property
    def entries(self):
        return self._plot_dict


def winsorize_col(col, upper_limit, lower_limit):
    return winsorize(col, limits=[upper_limit, lower_limit])


def map_y_range(a, b, Y):
    # TODO: could be smart to extract a as max and min from Y
    """
    Maps values from interval a[a1, a2] to interval b[b1, b2] for the given values of Y.
    Args:
    a (tuple): A tuple of two values (a1, a2) representing the lower and upper bounds of the input interval a.
    b (tuple): A tuple of two values (b1, b2) representing the lower and upper bounds of the output interval b.
    Y (list): A list of values to be mapped from interval a to interval b.
    Returns:
    A list of mapped values corresponding to the values in Y, with values in interval a[a1, a2] mapped to interval b[b1, b2].
    Example:
    >>> map_y_range((0, 10), (0, 1), [2, 5, 8])
    [0.2, 0.5, 0.8]
    """
    (a1, a2), (b1, b2) = a, b
    return [b1 + ((y - a1) * (b2 - b1) / (a2 - a1)) for y in Y]


def feature_importance_visualize(
    data_dict_global,
    folder="./results/",
    name="demo",
    save_png=False,
    save_eps=False,
    n_features=None,
):
    feature_importances = []
    feature_names = []
    for key, item in data_dict_global.items():
        if item["importance"] > 0:
            feature_importances.append(item["importance"])
            feature_names.append(key)

    feature_importances, feature_names = zip(
        *sorted(zip(feature_importances, feature_names), reverse=True)[:n_features]
    )

    if feature_importances:
        # fig = plt.figure()
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        fig.set_dpi(100)

        ax = plt.axes()
        ax.bar(np.arange(len(feature_importances)), feature_importances, color="grey")
        ax.set_xticks(np.arange(len(feature_importances)))
        ax.set_xticklabels(feature_names, rotation=90)
        plt.xlabel("Feature Name", fontsize=12)
        plt.ylim(0, np.max(feature_importances) + 0.05)
        plt.xlim(-1, len(feature_names))
        plt.title("Feature Importance")

        fig.tight_layout()
        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)
    plt.show()


# %%


def make_plot(
    X,
    Y,
    feature_name,
    model_name,
    dataset_name,
    scaler_dict,
    y_scaler,
    X_original,
    Ys_lower_bound=None,
    Ys_upper_bound=None,
    scale_y=False,
    distribution_plot=True,
):

    X = np.array(X)
    Y = np.array(Y)

    if Ys_lower_bound is not None and Ys_upper_bound is not None:
        Ys_lower_bound = np.array(Ys_lower_bound)
        Ys_upper_bound = np.array(Ys_upper_bound)

    if task == "regression":
        # inverse transform y (series) to original scale by using y_scaler
        Y = y_scaler.inverse_transform(Y.reshape(-1, 1)).squeeze()
        if Ys_lower_bound is not None and Ys_upper_bound is not None:
            Ys_lower_bound = y_scaler.inverse_transform(
                Ys_lower_bound.reshape(-1, 1)
            ).squeeze()
            Ys_upper_bound = y_scaler.inverse_transform(
                Ys_upper_bound.reshape(-1, 1)
            ).squeeze()

    Y = map_y_range((min(Y), max(Y)), (0, 100), Y) if scale_y else Y

    if Ys_lower_bound is not None and Ys_upper_bound is not None:
        Ys_lower_bound = (
            map_y_range(
                (min(Ys_lower_bound), max(Ys_lower_bound)), (0, 100), Ys_lower_bound
            )
            if scale_y
            else Ys_lower_bound
        )
        Ys_upper_bound = (
            map_y_range(
                (min(Ys_upper_bound), max(Ys_upper_bound)), (0, 100), Ys_upper_bound
            )
            if scale_y
            else Ys_upper_bound
        )

    # rescale numerical features
    if feature_name in dataset.numerical_cols and scaler_dict:
        X = scaler_dict[feature_name].inverse_transform(X.reshape(-1, 1)).squeeze()

    # de-normalize atemp and temp to have actual degree celcius
    if dataset_name == "bike" and (feature_name == "atemp" or feature_name == "temp"):
        if feature_name == "atemp":
            X = X * (50 - (-16)) + (-16)
        if feature_name == "temp":
            X = X * (39 - (-8)) + (-8)

    # create distribution plot
    if distribution_plot:
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, gridspec_kw={"height_ratios": [0.8, 0.2]}
        )
        if feature_name in dataset.numerical_cols:
            bins_values, _, _ = ax2.hist(
                X_original[feature_name], bins=10, rwidth=0.9, color="grey"
            )
        else:
            bins_values, _, _ = ax2.hist(
                X[feature_name], bins=10, rwidth=0.9, color="grey"
            )
        ax2.set_xlabel("Distribution")
        ax2.set_xticks([])
        ax2.set_yticks([0, max(bins_values)])

    else:
        fig, ax1 = plt.subplots(nrows=1)

    fig.set_size_inches(5, 5)
    fig.set_dpi(100)

    if model_name != "EBM":
        ax1.plot(X, Y, color="black", alpha=1)
        if Ys_lower_bound is not None and Ys_upper_bound is not None:
            ax1.fill_between(
                X,
                Ys_lower_bound,
                Ys_upper_bound,
                color="grey",
                alpha=0.2,
                label="Confidence Band",
            )
    else:
        ax1.step(X, Y, where="post", color="black")
        if Ys_lower_bound is not None and Ys_upper_bound is not None:
            ax1.fill_between(
                X,
                Ys_lower_bound,
                Ys_upper_bound,
                step="post",
                color="grey",
                alpha=0.2,
                label="Confidence Band",
            )
    ax1.set_title(f"Feature:{feature_name}")
    ax1.set_xlabel(f"Feature value")
    ax1.set_ylabel("Feature effect on model output")
    fig.tight_layout()
    plt.savefig(f"plots/{dataset_name}/{model_name}_shape_{feature_name}.png", dpi=300)
    plt.show()
    plt.close(fig)


# def plot_continuous_bar(
#     data_dict, feature_name, model_name, dataset_name, multiclass=False, show_error=True, title=None, xtitle="", ytitle=""
# ):
#     if feature_name == "capital.gain":
#         print("BUG")
#
#     if data_dict.get("scores", None) is None:  # pragma: no cover
#         return None
#
#     x_vals = data_dict["names"].copy()
#     y_vals = data_dict["scores"].copy()
#     y_hi = data_dict.get("upper_bounds", None)
#     y_lo = data_dict.get("lower_bounds", None)
#
#     # x_min = min(x_vals)
#     # x_max = max(x_vals)
#
#     if y_hi is None or multiclass:
#         show_error = False
#
#     def extend_x_range(x):
#         return x
#
#     def extend_y_range(y):
#         return np.r_[y, y[np.newaxis, -1]]
#
#     new_x_vals = extend_x_range(x_vals)
#     new_y_vals = extend_y_range(y_vals)
#     if show_error:
#         new_y_hi = extend_y_range(y_hi)
#         new_y_lo = extend_y_range(y_lo)
#
#     data = []
#     fill = "none"
#     if show_error:
#         fill = "tonexty"
#
#     if multiclass:
#         for i in range(y_vals.shape[1]):
#             class_name = (
#                 "Class {}".format(i)
#                 if "meta" not in data_dict
#                 else data_dict["meta"]["label_names"][i]
#             )
#             class_line = go.Scatter(
#                 x=new_x_vals,
#                 y=new_y_vals[:, i],
#                 line=dict(shape="hv"),
#                 name=class_name,
#                 mode="lines",
#             )
#             data.append(class_line)
#     else:
#         main_line = go.Scatter(
#             x=new_x_vals,
#             y=new_y_vals,
#             name="Main",
#             mode="lines",
#             line=dict(color="rgb(31, 119, 180)", shape="hv"),
#             fillcolor="rgba(68, 68, 68, 0.15)",
#             fill=fill,
#         )
#         data.append(main_line)
#
#     if show_error:
#         upper_bound = go.Scatter(
#             name="Upper Bound",
#             x=new_x_vals,
#             y=new_y_hi,
#             mode="lines",
#             marker=dict(color="#444"),
#             line=dict(width=0, shape="hv"),
#             fillcolor="rgba(68, 68, 68, 0.15)",
#             fill="tonexty",
#         )
#         lower_bound = go.Scatter(
#             name="Lower Bound",
#             x=new_x_vals,
#             y=new_y_lo,
#             marker=dict(color="#444"),
#             line=dict(width=0, shape="hv"),
#             mode="lines",
#         )
#         data = [lower_bound, main_line, upper_bound]
#
#     show_legend = True if multiclass or not show_error else False
#     layout = go.Layout(
#         title=title,
#         showlegend=show_legend,
#         xaxis=dict(title=xtitle),
#         yaxis=dict(title=ytitle),
#     )
#     yrange = None
#     if data_dict.get("scores_range", None) is not None:
#         scores_range = data_dict["scores_range"]
#         yrange = scores_range
#
#     main_fig = go.Figure(data=data, layout=layout)
#     main_fig.show()
#     main_fig.write_image(f'plots/{model_name}_{dataset_name}_shape_{feature_name}.pdf')


# def make_plot_ebm(data_dict, feature_name, model_name, dataset_name, num_epochs='', debug=False):
#     x_vals = data_dict["names"].copy()
#     y_vals = data_dict["scores"].copy()
#
#     # This is important since you do not plot plt.stairs with len(edges) == len(vals) + 1, which will have a drop to zero at the end
#     y_vals = np.r_[y_vals, y_vals[np.newaxis, -1]]
#
#     # This is the code interpretml also uses: https://github.com/interpretml/interpret/blob/2327384678bd365b2c22e014f8591e6ea656263a/python/interpret-core/interpret/visual/plot.py#L115
#
#     # main_line = go.Scatter(
#     #     x=x_vals,
#     #     y=y_vals,
#     #     name="Main",
#     #     mode="lines",
#     #     line=dict(color="rgb(31, 119, 180)", shape="hv"),
#     #     fillcolor="rgba(68, 68, 68, 0.15)",
#     #     fill="none",
#     # )
#     #
#     # main_fig = go.Figure(data=[main_line])
#     # main_fig.show()
#     # main_fig.write_image(f'plots/{model_name}_{dataset_name}_shape_{feature_name}_{num_epochs}epochs.pdf')
#
#     # This is my custom code used for plotting
#
#     x = np.array(x_vals)
#     if debug:
#         print("Num cols:", dataset.numerical_cols)
#     if feature_name in dataset.numerical_cols:
#         if debug:
#             print("Feature to scale back:", feature_name)
#         x = scaler_dict[feature_name].inverse_transform(x.reshape(-1, 1)).squeeze()
#     else:
#         if debug:
#             print("Feature not to scale back:", feature_name)
#
#     fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [0.8, 0.2]})
#     fig.set_size_inches(5, 5)
#     fig.set_dpi(100)
#
#     ax1.step(x, y_vals, where="post", color='black')
#     bins_values, _, _ = ax2.hist(X_original[feature_name], bins=10, rwidth=0.9, color='grey')
#
#     # ax1.legend(loc='best')
#     ax1.set_title(f'Feature:{feature_name}')
#     ax1.set_xlabel(f'Feature value')
#     ax1.set_ylabel('Feature effect on model output')
#     ax2.set_xlabel("Distribution")
#     ax2.set_xticks([])
#     ax2.set_yticks([0, max(bins_values)])
#
#     # plt.xlabel(f'Feature value')
#     # plt.ylabel('Feature effect on model output')
#     # plt.title(f'Feature:{feature_name}')
#     fig.tight_layout()
#     plt.savefig(f'plots/{model_name}_{dataset_name}_shape_{feature_name}_{num_epochs}epochs.pdf')
#     plt.show()


def make_plot_interaction(
    left_names,
    right_names,
    scores,
    feature_name_left,
    feature_name_right,
    model_name,
    dataset_name,
    scale_back=True,
):
    """
    This function is used to plot the interaction plot in a heatmap style.
    It is used by EBM and Gaminet.

    :param left_names:
    :param right_names:
    :param scores:
    :param feature_name_left:
    :param feature_name_right:
    :param model_name:
    :param dataset_name:
    :param scale_back:
    :return:
    """

    left_names = np.array(left_names)
    right_names = np.array(right_names)

    if feature_name_left in dataset.numerical_cols and scale_back:
        left_names = (
            scaler_dict[feature_name_left]
            .inverse_transform(left_names.reshape(-1, 1))
            .squeeze()
        )
    if feature_name_right in dataset.numerical_cols and scale_back:
        right_names = (
            scaler_dict[feature_name_right]
            .inverse_transform(right_names.reshape(-1, 1))
            .squeeze()
        )

    if "_" in feature_name_right:
        right_names = right_names.astype("str")

    if "_" in feature_name_left:
        left_names = left_names.astype("str")

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    fig.set_dpi(100)

    if task == "regression":
        # inverse transform y (series) to original scale by using y_scaler
        scores = y_scaler.inverse_transform(scores)
    im = ax.pcolormesh(left_names, right_names, scores, shading="auto")
    fig.colorbar(im, ax=ax)
    plt.xlabel(feature_name_left)
    plt.ylabel(feature_name_right)
    plt.title(
        f'Feature: {feature_name_left.replace("?", "missing")} x {feature_name_right.replace("?", "missing")}'
    )
    fig.tight_layout()
    plt.savefig(
        f'plots/{dataset_name}/{model_name}_interact_{feature_name_left.replace("?", "missing")} x {feature_name_right.replace("?", "missing")}.pdf'
    )
    plt.show()


def make_plot_interaction_onehot_line_gaminet(
    categorical_input,
    continuous_input,
    output,
    categorical_name,
    continuous_name,
    model_name,
    dataset_name,
):
    """

    :param categorical_input:
    :param continuous_input:
    :param output:
    :param categorical_name:
    :param continuous_name:
    :param model_name:
    :param dataset_name:
    :return:
    """
    categorical_input = np.array(categorical_input)
    continuous_input = np.array(continuous_input)

    if "_" in continuous_name:
        continuous_input = continuous_input.astype(
            "str"
        )  # since we want categorical values and not a range
    bin_vals = output.transpose(-1, 0)
    bin_vals = np.ascontiguousarray(bin_vals)

    bin_vals_y_0 = bin_vals[0]
    bin_vals_y_1 = bin_vals[1]
    if task == "regression":
        # inverse transform y (series) to original scale by using y_scaler
        bin_vals_y_0 = y_scaler.inverse_transform(bin_vals_y_0.reshape(-1, 1))
        bin_vals_y_1 = y_scaler.inverse_transform(bin_vals_y_1.reshape(-1, 1))

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    fig.set_dpi(100)

    plt.plot(
        continuous_input,
        bin_vals_y_0,
        color="black",
        label=f"{categorical_name} = {int(categorical_input[0])}",
    )  # y_vals[0]
    plt.plot(
        continuous_input,
        bin_vals_y_1,
        color="grey",
        label=f"{categorical_name} = {int(categorical_input[1])}",
    )  # y_vals[1]
    plt.legend(loc="best")
    # plt.fill_between(x, lower_bounds, mean, color='gray')
    # plt.fill_between(x, mean, upper_bounds, color='gray')
    plt.xlabel(f"Feature value")
    plt.ylabel("Feature effect on model output")
    plt.title(f"Feature:{categorical_name} x {continuous_name}")
    fig.tight_layout()
    plt.savefig(
        f"plots/{dataset_name}/{model_name}_interact_{categorical_name} x {continuous_name}.pdf"
    )
    plt.show()


def make_plot_interaction_onehot_line_ebm(
    data_dict, categorical_name, continuous_name, model_name, dataset_name
):
    x_positions = np.array(data_dict["right_names"])
    categories = np.array(data_dict["left_names"])
    bin_vals = data_dict["scores"]
    bin_vals = np.ascontiguousarray(bin_vals)

    bin_vals_y_0 = bin_vals[0]
    bin_vals_y_1 = bin_vals[1]
    if task == "regression":
        # inverse transform y (series) to original scale by using y_scaler
        bin_vals_y_0 = y_scaler.inverse_transform(bin_vals_y_0.reshape(-1, 1)).squeeze()
        bin_vals_y_1 = y_scaler.inverse_transform(bin_vals_y_1.reshape(-1, 1)).squeeze()

    bin_vals_y_0 = np.r_[bin_vals_y_0, bin_vals_y_0[np.newaxis, -1]]
    bin_vals_y_1 = np.r_[bin_vals_y_1, bin_vals_y_1[np.newaxis, -1]]

    if continuous_name in dataset.numerical_cols:
        x_positions = (
            scaler_dict[continuous_name]
            .inverse_transform(x_positions.reshape(-1, 1))
            .squeeze()
        )
        # inverse_transform x_positions with the one hot encoder

    fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [0.8, 0.2]})
    fig.set_size_inches(5, 5)
    fig.set_dpi(100)

    ax1.step(
        x_positions,
        bin_vals_y_0,
        where="post",
        color="grey",
        label=f"{categorical_name} = {int(float(categories[0]))}",
    )  # y_vals[0]
    ax1.step(
        x_positions,
        bin_vals_y_1,
        where="post",
        color="black",
        label=f"{categorical_name} = {int(float(categories[1]))}",
    )  # y_vals[1]
    bins_values, _, _ = ax2.hist(
        X_original[continuous_name], align="mid", bins=10, rwidth=0.9, color="grey"
    )

    ax1.legend(loc="best")
    ax1.set_title(f"Feature:{categorical_name} x {continuous_name}")
    ax1.set_xlabel(f"Feature value: {continuous_name}")
    ax1.set_ylabel("Feature effect on model output")
    ax2.set_xlabel("Distribution")
    ax2.set_xticks([])
    ax2.set_yticks([0, max(bins_values)])
    fig.tight_layout()

    # fig.title(f'Feature:{categorical_name} x {continuous_name}')
    plt.savefig(
        f"plots/{dataset_name}/{model_name}_interact_{categorical_name} x {continuous_name}.pdf"
    )
    plt.show()


def plot_pairwise_heatmap(
    data_dict, x_name, y_name, model_name, dataset_name, title="", xtitle="", ytitle=""
):

    if len(data_dict.get("scores", [])) != 2:
        warnings.warn(
            "Only 2 classes supported for now. The data_dict in plot_pairwise_heatmap should have exactly 2 classes. No plot will be generated."
        )
        return

    if x_name in dataset.numerical_cols or y_name in dataset.numerical_cols:
        warnings.warn(
            "plot_pairwise_heatmap() is ment to be used only for cat x cat interactions. One of the plot features is actually numerical. No plot will be generated."
        )
        return

    bin_vals = data_dict["scores"]
    # Has to be transposed so the heatmap will be shown correctly using pcolormesh
    bin_vals = np.ascontiguousarray(np.transpose(bin_vals, (1, 0)))
    if task == "regression":
        # inverse transform y (series) to original scale by using y_scaler
        bin_vals = y_scaler.inverse_transform(bin_vals.reshape(-1, 1))

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    fig.set_dpi(100)
    ax.tick_params(axis="both", which="major", labelsize=8)

    im = ax.pcolormesh(bin_vals, shading="auto")
    fig.colorbar(im, ax=ax)
    bin_labels_x = [
        f'{x_name.split("_")[1]} not given',
        f'{x_name.split("_")[1]} given',
    ]
    bin_labels_y = [
        f'{y_name.split("_")[1]} not given',
        f'{y_name.split("_")[1]} given',
    ]
    plt.xticks([0.5, 1.5], bin_labels_x)
    plt.yticks([0.5, 1.5], bin_labels_y, rotation=90, va="center")

    plt.title(f'Feature: {x_name.replace("?", "")} x {y_name.replace("?", "")}')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    fig.tight_layout()
    plt.savefig(
        f'plots/{dataset_name}/{model_name}_interact_{x_name.replace("?", "")} x {y_name.replace("?", "")}.pdf'
    )
    plt.show()


def make_one_hot_plot(
    class_zero, class_one, feature_name, model_name, dataset_name, y_scaler, X_original
):
    # This snippet is for binary plots
    original_feature_name = feature_name.split("_")[0]
    if task == "regression":
        # inverse transform y (series) to original scale by using y_scaler
        class_zero = y_scaler.inverse_transform(class_zero.reshape(1, -1)).item()
        class_one = y_scaler.inverse_transform(class_one.reshape(1, -1)).item()

    fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [0.8, 0.2]})
    fig.set_size_inches(5, 5)
    fig.set_dpi(100)

    """
    if X_original[original_feature_name].value_counts().size == 2:
        category_0 = X_original[original_feature_name].values.categories[0]
        category_1 = X_original[original_feature_name].values.categories[1]
        categories = [category_0, category_1]
        ax1.bar([0, 1], [class_zero, class_one], color='gray', tick_label=[f'{categories[0]}', f'{categories[1]} '])

    # This snippet is for all plots but binary categories
    # TODO: Binary-Class One Hot Plot für Multi-Class Plots rausnehmen
    else:
        ax1.bar([0, 1], [class_zero, class_one], color='gray',
                tick_label=[f'{feature_name} = 0', f'{feature_name} = 1'])
    """

    bins_values, _, _ = ax2.hist(
        X_original[original_feature_name], bins=2, rwidth=0.9, color="grey"
    )
    plt.title(f'Feature:{feature_name.split("_")[0]}')
    ax1.set_ylabel("Feature effect on model output")
    ax2.set_xlabel("Distribution")
    ax2.set_xticks([])
    ax2.set_yticks([0, max(bins_values)])

    fig.tight_layout()
    plt.savefig(
        f'plots/{dataset_name}/{model_name}_onehot_{str(feature_name).replace("?", "")}.pdf'
    )
    plt.show()


def make_one_hot_multi_plot(
    plot_data: PlotData,
    model_name,
    dataset_name,
    y_scaler,
    X_original,
    distibution_plot=True,
):
    for feature_name in plot_data.entries:
        position_list = np.arange(len(plot_data.entries[feature_name]))
        y_values = list(plot_data.entries[feature_name].values())
        y_list_not_given_class = [
            list(dict_element.values())[0] for dict_element in y_values
        ]
        y_list_given_class = [
            list(dict_element.values())[1] for dict_element in y_values
        ]
        if task == "regression":
            # inverse transform y (series) to original scale by using y_scaler
            y_list_not_given_class = y_scaler.inverse_transform(
                np.array(y_list_not_given_class).reshape((-1, 1))
            ).squeeze()
            y_list_given_class = y_scaler.inverse_transform(
                np.array(y_list_given_class).reshape((-1, 1))
            ).squeeze()

        x_list = list(plot_data.entries[feature_name].keys())

        if distibution_plot:
            fig, (ax1, ax2) = plt.subplots(
                nrows=2, gridspec_kw={"height_ratios": [0.8, 0.2]}
            )
            bins_values, _, _ = ax2.hist(
                X_original[feature_name], bins=len(x_list), rwidth=0.8, color="grey"
            )
            ax2.set_xlabel("Distribution")
            ax2.set_xticks([])
            ax2.set_yticks([0, max(bins_values)])
        else:
            fig, ax1 = plt.subplots()
        fig.set_size_inches(5, 5)
        fig.set_dpi(100)

        # The for loop is used to calculate the y values for the plot (add up the given value and all not given values of the other classes)
        # e.g. for class 1: add given value of class 1 + not_given_value class 0 + not_given_value class 2 + not_given_value class 3
        y_plot_value = []
        for i in range(len(y_values)):
            y_not_given_values = sum(
                [
                    value
                    for index, value in enumerate(y_list_not_given_class)
                    if index != i
                ]
            )
            y_plot_value.append((y_list_given_class[i] + y_not_given_values).item())

        ax1.bar(position_list, y_plot_value, color="gray", width=0.8)

        # Relict of showing y_list_not_given_class and y_list_given_class seperately
        # ax1.bar(position_list-0.2, y_list_not_given_class, color='silver', width=0.4)
        # ax1.bar(position_list+0.2, y_list_given_class, color='gray', width=0.4)
        # ax1.legend(['Not given', 'Given'])
        ax1.set_ylabel("Feature effect on model output")
        ax1.set_title(f"Feature:{feature_name}")
        ax1.set_xticks(position_list)
        ax1.set_xticklabels(x_list, rotation=90)
        fig.tight_layout()
        plt.savefig(
            f'plots/{dataset_name}/{model_name}_multi_onehot_{str(feature_name).replace("?", "")}.pdf',
            bbox_inches="tight",
        )
        plt.show()


# %%


def EBM_show(X, y):
    m4 = ExplainableBoostingRegressor(interactions=10, max_bins=256)
    m4.fit(X, y)
    ebm_global = m4.explain_global()
    show(ebm_global)


def EBM(dataset, feature_name, model_name="EBM"):
    Xs = []
    Ys = []
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        X, y, X_original, y_scaler, scaler_dict = transform_dataset(
            dataset=dataset, random_state=seed, split=True
        )
        if task == "classification":
            ebm = ExplainableBoostingClassifier(
                interactions=10, max_bins=256, outer_bags=8, inner_bags=0
            )
        else:
            ebm = ExplainableBoostingRegressor(
                interactions=10, max_bins=256, outer_bags=8, inner_bags=0
            )
        ebm.fit(X, y)
        ebm_global = ebm.explain_global()
        for i, shape_name in enumerate(
            ebm_global.data()["names"]
        ):  # shape_name besserer Name?
            if shape_name == feature_name:
                shape_data = ebm_global.data(i)
                Xs.append(shape_data["names"].copy())
                Ys.append(shape_data["scores"].copy())
            # data_names = ebm_global.data()
            # feature_name = data_names["names"][i]

            # print(feature_name)

            # X_values = shape_data["names"].copy()
            # Y_values = shape_data["scores"].copy()
            # Y_values = np.r_[Y_values, Y_values[np.newaxis, -1]]

    min_length = min(len(arr) for arr in Ys)
    Ys = [arr[:min_length] for arr in Ys]
    Xs = [arr[:min_length] for arr in Xs]

    Ys_lower_bound = np.mean(Ys, axis=0) - np.std(Ys, axis=0)
    Ys_upper_bound = np.mean(Ys, axis=0) + np.std(Ys, axis=0)
    X = np.mean(Xs, axis=0)
    Y = np.mean(Ys, axis=0)

    make_plot(
        X,
        Y,
        feature_name,
        model_name,
        dataset.name,
        scaler_dict,
        y_scaler,
        X_original,
        Ys_lower_bound,
        Ys_upper_bound,
    )


def PYGAM(dataset, feature_name, model_name="PYGAM"):
    Xs = []
    Ys = []
    for seed in [1]:
        model_is_fitted = False
        while model_is_fitted is not True:
            X, y, X_original, y_scaler, scaler_dict = transform_dataset(
                dataset=dataset, random_state=seed, split=True
            )

            # TODO: Integrate terms as parameters on model initialization
            tms = terms.TermList(
                *[
                    (
                        f(i)
                        if X.columns[i] in dataset.categorical_cols
                        else s(i, n_splines=20, lam=0.6)
                    )
                    for i in range(X.shape[1])
                ]
            )

            try:
                # fit the model
                if task == "classification":
                    PYGAM = LogisticGAM(tms).fit(X, y)
                elif task == "regression":
                    PYGAM = LinearGAM(tms).fit(X, y)
                model_is_fitted = True
            except:
                message = f"For this seed: {seed} no fit was found."
                print(f"\033[91m{message}\033[0m")
                seed = seed + 10
                print(f"\033[0;32m{f'New seed is: {seed}!'}\033[0m")

        for i, term in enumerate(PYGAM.terms):
            if term.isintercept:
                continue
            XX = PYGAM.generate_X_grid(term=i)
            pdep, confi = PYGAM.partial_dependence(term=i, X=XX, width=0.95)
            # make_plot(XX[:,i].squeeze(), pdep, confi[:,0], confi[:,1], X.columns[i])
            original_feature_name = X[X.columns[i]].name.split("_")[0]
            if (
                original_feature_name == feature_name
                or original_feature_name == "targeted"
            ):
                Xs.append(XX[:, i].squeeze())
                Ys.append(pdep)

    min_length = min(len(arr) for arr in Ys)
    Ys = [arr[:min_length] for arr in Ys]
    Xs = [arr[:min_length] for arr in Xs]

    Ys_lower_bound = np.mean(Ys, axis=0) - np.std(Ys, axis=0)
    Ys_upper_bound = np.mean(Ys, axis=0) + np.std(Ys, axis=0)
    X = np.mean(Xs, axis=0)
    Y = np.mean(Ys, axis=0)

    # X_no_split = np.array(X_no_split)
    # Y_no_split = np.array(Y_no_split)
    # Ys = np.array(feature_Ys[feature_name])
    # Ys = Ys.transpose(1, 0)
    # Ys_upper_bound = []
    # Ys_lower_bound = []

    # for i, _ in enumerate(Ys):
    #     Ys_lower_bound.append(np.mean(Ys[i]) - np.std(Ys[i]))
    #     Ys_upper_bound.append(np.mean(Ys[i]) + np.std(Ys[i]))

    make_plot(
        X,
        Y,
        feature_name,
        model_name,
        dataset.name,
        scaler_dict,
        y_scaler,
        X_original,
        Ys_lower_bound,
        Ys_upper_bound,
    )


def Gaminet(dataset, feature_name, model_name="Gaminet"):
    X_no_split = []
    Xs = []
    Ys = []
    for seed in [1]:
        X, y, X_original, y_scaler, scaler_dict = transform_dataset(
            dataset=dataset, random_state=seed, split=False
        )

        x_types = {}
        for i in range(len(X.columns)):
            if "_" in X.columns[i]:
                x_types[X.columns[i]] = {"type": "categorical", "values": [0, 1]}
            else:
                x_types[X.columns[i]] = {"type": "continuous"}

        # meta_info = {X.columns[i]: {'type': 'continuous'} for i in range(len(X.columns))}
        meta_info = x_types
        meta_info.update({"Y": {"type": "target"}})

        # from sklearn.preprocessing import FunctionTransformer
        identity = FunctionTransformer()

        for i, (key, item) in enumerate(meta_info.items()):
            if item["type"] == "target":
                continue
            # sx = MinMaxScaler((0, 1))
            # sx.fit([[0], [1]])
            # print(scaler_dict.keys())
            # print(X.columns)
            if key in scaler_dict and item["type"] != "categorical":
                meta_info[key]["scaler"] = scaler_dict[key]
            else:
                meta_info[key]["scaler"] = identity

        if task == "classification":  # 190 it/ s Maineffect, 65 it/s Interaction
            model_to_run = GAMINetClassifier(
                batch_size=1024,
                interact_num=20,
                activation_func="ReLU",
                reg_clarity=0.1,
                warm_start=False,
                max_epochs=(3000, 1000, 1000),
                verbose=True,
                device="cpu",  # cuda takes est. 2-3x longer
                random_state=0,
            )
            model_to_run.fit(np.array(X), np.array(y).reshape(-1, 1))

        elif task == "regression":
            model_to_run = GAMINetRegressor(
                batch_size=1024,
                interact_num=20,
                activation_func="ReLU",
                reg_clarity=0.1,
                warm_start=False,
                max_epochs=(3000, 1000, 1000),
                verbose=True,
                device="cpu",  # cuda takes est. 3x longer
                random_state=0,
            )
            model_to_run.fit(np.array(X), np.array(y).reshape(-1, 1))

        data_dict = model_to_run.global_explain(main_grid_size=256)

        Xnames2Featurenames = dict(
            zip([X.columns[i] for i in range(X.shape[1])], X.columns)
        )

        feature_index = list(Xnames2Featurenames).index(feature_name)
        Xs.append(data_dict[f"X{feature_index}"]["inputs"])
        Ys.append(data_dict[f"X{feature_index}"]["outputs"])

    Ys_lower_bound = np.mean(Ys, axis=0) - np.std(Ys, axis=0)
    Ys_upper_bound = np.mean(Ys, axis=0) + np.std(Ys, axis=0)
    X = np.mean(Xs, axis=0)
    Y = np.mean(Ys, axis=0)

    make_plot(
        X,
        Y,
        feature_name,
        model_name,
        dataset.name,
        scaler_dict,
        y_scaler,
        X_original,
        Ys_lower_bound,
        Ys_upper_bound,
    )

    # for effect in data_dict.keys():
    #     if data_dict[effect]["type"] == "continuous":
    #         make_plot(
    #             data_dict[effect]["inputs"],
    #             data_dict[effect]["outputs"],
    #             Xnames2Featurenames[effect],
    #             model_name,
    #             dataset_name,
    #             scale_back=False,
    #         )


def Gaminet_complete(dataset, feature_name, model_name="Gaminet"):
    X, y, X_original, y_scaler, scaler_dict = transform_dataset(
        dataset=dataset, random_state=1, split=False
    )

    x_types = {}
    for i in range(len(X.columns)):
        if "_" in X.columns[i]:
            x_types[X.columns[i]] = {"type": "categorical", "values": [0, 1]}
        else:
            x_types[X.columns[i]] = {"type": "continuous"}

    # meta_info = {X.columns[i]: {'type': 'continuous'} for i in range(len(X.columns))}
    meta_info = x_types
    meta_info.update({"Y": {"type": "target"}})

    # from sklearn.preprocessing import FunctionTransformer
    identity = FunctionTransformer()

    for i, (key, item) in enumerate(meta_info.items()):
        if item["type"] == "target":
            continue
        # sx = MinMaxScaler((0, 1))
        # sx.fit([[0], [1]])
        # print(scaler_dict.keys())
        # print(X.columns)
        if key in scaler_dict and item["type"] != "categorical":
            meta_info[key]["scaler"] = scaler_dict[key]
        else:
            meta_info[key]["scaler"] = identity

    if task == "classification":  # 190 it/ s Maineffect, 65 it/s Interaction
        model_to_run = GAMINetClassifier(
            feature_names=list(X.columns),
            batch_size=1024,
            interact_num=20,
            activation_func="ReLU",
            reg_clarity=0.1,
            warm_start=False,
            max_epochs=(3000, 1000, 1000),
            verbose=True,
            device="cpu",  # cuda takes est. 2-3x longer
            random_state=0,
        )
        model_to_run.fit(np.array(X), np.array(y).reshape(-1, 1))

    elif task == "regression":
        model_to_run = GAMINetRegressor(
            feature_names=list(X.columns),
            batch_size=1024,
            interact_num=20,
            activation_func="ReLU",
            reg_clarity=0.1,
            warm_start=False,
            max_epochs=(3000, 1000, 1000),
            verbose=True,
            device="cpu",  # cuda takes est. 3x longer
            random_state=0,
        )
        model_to_run.fit(np.array(X), np.array(y).reshape(-1, 1))

    data_dict = model_to_run.global_explain(main_grid_size=256)

    make_plot(
        data_dict[feature_name]["inputs"],
        data_dict[feature_name]["outputs"],
        feature_name,
        model_name,
        dataset.name,
        scaler_dict,
        y_scaler,
        X_original,
    )


def EXNN(X, y, dataset_name, model_name="ExNN"):
    meta_info = {
        "X" + str(i + 1): {"type": "continuous"} for i in range(len(X.columns))
    }
    meta_info.update({"Y": {"type": "target"}})

    for i, (key, item) in enumerate(meta_info.items()):
        if item["type"] == "target":
            continue
        sx = MinMaxScaler((0, 1))
        sx.fit([[0], [1]])
        meta_info[key]["scaler"] = sx

    X_arr = np.array(X)
    y_arr = np.array(y)

    if task == "classification":
        model_to_run = ExNN(
            meta_info=meta_info,
            subnet_num=10,
            subnet_arch=[10, 6],
            task_type="Classification",
            activation_func=tf.tanh,
            batch_size=min(1000, int(X.shape[0] * 0.2)),
            training_epochs=10000,  # default 10000
            lr_bp=0.001,
            lr_cl=0.1,
            beta_threshold=0.05,
            tuning_epochs=100,
            l1_proj=0.0001,
            l1_subnet=0.00316,
            l2_smooth=10 ** (-6),
            verbose=True,
            val_ratio=0.2,
            early_stop_thres=500,
        )
        model_to_run.fit(X_arr, y_arr)

        model_to_run.visualize(
            save_png=True, folder="plots/", name=f"{model_name}_{dataset_name}_shape"
        )

    elif task == "regression":
        model_to_run = ExNN(
            meta_info=meta_info,
            subnet_num=10,
            subnet_arch=[10, 6],
            task_type="Regression",
            activation_func=tf.tanh,
            batch_size=min(1000, int(X.shape[0] * 0.2)),
            training_epochs=10000,  # default
            lr_bp=0.001,
            lr_cl=0.1,
            beta_threshold=0.05,
            tuning_epochs=100,
            l1_proj=0.0001,
            l1_subnet=0.00316,
            l2_smooth=10 ** (-6),
            verbose=True,
            val_ratio=0.2,
            early_stop_thres=500,
        )
        model_to_run.fit(X_arr, y_arr)

        model_to_run.visualize(
            save_png=True, folder="plots/", name=f"{model_name}_{dataset_name}_shape"
        )


def LR(dataset, feature_name, model_name="LR"):
    Xs = []
    Ys = []
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        X, y, X_original, y_scaler, scaler_dict = transform_dataset(
            dataset=dataset, random_state=seed, split=True
        )
        if dataset.problem == "regression":
            m = ElasticNet(alpha=0.001, l1_ratio=0)
        else:
            m = LogisticRegression(
                C=0.1,
                penalty="l1",
                class_weight=None,
                solver="liblinear",
                max_iter=100,
                n_jobs=-1,
            )
        m.fit(X, y)

        word_to_coef = dict(zip(m.feature_names_in_, m.coef_.squeeze()))
        dict(sorted(word_to_coef.items(), key=lambda item: item[1]))

        # for i, feature in enumerate(X.columns):
        #     original_feature_name = feature.split("_")[0]
        #     if original_feature_name == feature_name:

        inp = np.linspace(X[feature_name].min(), X[feature_name].max(), 1000)
        outp = inp * word_to_coef[feature_name] + m.intercept_

        Xs.append(inp)
        Ys.append(outp)

    Ys_lower_bound = np.mean(Ys, axis=0) - np.std(Ys, axis=0)
    Ys_upper_bound = np.mean(Ys, axis=0) + np.std(Ys, axis=0)
    X = np.mean(Xs, axis=0)
    Y = np.mean(Ys, axis=0)

    # Ys = np.array(feature_Ys[feature_name])
    # min_length = min(len(arr) for arr in Ys)
    # trimmed_arrays = [arr[:min_length] for arr in Ys]

    # Stack the trimmed arrays into a single NumPy array
    # Ys = np.array(trimmed_arrays).transpose(1, 0)
    # Ys_upper_bound = []
    # Ys_lower_bound = []

    # for i, _ in enumerate(Ys):
    #    Ys_lower_bound.append(np.mean(Ys[i]) - np.std(Ys[i]))
    #    Ys_upper_bound.append(np.mean(Ys[i]) + np.std(Ys[i]))

    # if len(Ys_lower_bound) < len(X_no_split):
    #    X_no_split = X_no_split[: len(Ys_lower_bound)]
    #    Y_no_split = Y_no_split[: len(Ys_lower_bound)]

    make_plot(
        X,
        Y,
        feature_name,
        model_name,
        dataset.name,
        scaler_dict,
        y_scaler,
        X_original,
        Ys_lower_bound,
        Ys_upper_bound,
    )

    # make_plot(
    #     inp,
    #     outp,
    #     feature_name,
    #     model_name=model_name,
    #     dataset_name=dataset_name,
    # )


def I_GANN(dataset, feature_name, model_name="IGANN"):

    Xs = []
    Ys = []
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        X, y, X_original, y_scaler, scaler_dict = transform_dataset(
            dataset=dataset, random_state=seed, split=True
        )
        # m = IGANN(task=dataset.problem, verbose=1)
        m = IGANN(
            task=dataset.problem, boost_rate=0.1, elm_scale=1, interactions=0, verbose=1
        )
        m.fit(X, y)

        shape_data = m.get_shape_functions_as_dict()
        for feature in shape_data:
            original_feature_name = feature["name"].split("_")[0]
            if (
                original_feature_name == feature_name
                or original_feature_name == "targeted"
            ):
                Xs.append(feature["x"].numpy())
                Ys.append(feature["y"].numpy())
                # if X_original[original_feature_name].value_counts().size > 2:
                #     # print(feature)
                #     column_name = original_feature_name
                #     class_name = feature["name"].split("_")[1]
    min_length = min(len(arr) for arr in Ys)
    Ys = [arr[:min_length] for arr in Ys]
    Xs = [arr[:min_length] for arr in Xs]

    Ys_lower_bound = np.mean(Ys, axis=0) - np.std(Ys, axis=0)
    Ys_upper_bound = np.mean(Ys, axis=0) + np.std(Ys, axis=0)
    X = np.mean(Xs, axis=0)
    Y = np.mean(Ys, axis=0)
    # Ys = np.array(feature_Ys[feature_name])
    # min_length = min(len(arr) for arr in Ys)
    # trimmed_arrays = [arr[:min_length] for arr in Ys]

    # # Stack the trimmed arrays into a single NumPy array
    # Ys = np.array(trimmed_arrays).transpose(1, 0)
    # Ys_upper_bound = []
    # Ys_lower_bound = []

    # for i, _ in enumerate(Ys):
    #     Ys_lower_bound.append(np.mean(Ys[i]) - np.std(Ys[i]))
    #     Ys_upper_bound.append(np.mean(Ys[i]) + np.std(Ys[i]))

    # if len(Ys_lower_bound) < len(X_no_split):
    #     X_no_split = X_no_split[: len(Ys_lower_bound)]
    #     Y_no_split = Y_no_split[: len(Ys_lower_bound)]

    make_plot(
        X,
        Y,
        feature_name,
        model_name,
        dataset.name,
        scaler_dict,
        y_scaler,
        X_original,
        Ys_lower_bound,
        Ys_upper_bound,
    )


def MGCV(dataset, feature_name, model_name="MGCV"):
    pandas2ri.activate()
    utils = packages.importr("utils")
    utils.chooseCRANmirror(ind=1)
    stats = packages.importr("stats")
    base = packages.importr("base")
    mgcv_ = utils.install_packages("mgcv")
    mgcv = packages.importr("mgcv")

    Xs = []
    Ys = []
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        model_is_fitted = False
        while model_is_fitted is not True:
            try:
                X, y, X_original, y_scaler, scaler_dict = transform_dataset(
                    dataset=dataset, random_state=seed, split=True
                )

                if task == "classification":
                    m = MyRSplineClassifier(
                        random_state=1337,
                        maxk=10,
                        m=2,
                        gamma=1,
                        model_to_use="gam",
                        # model_to_use=self.arg_dict["model_to_use"],
                        spline_type="ts",
                        discrete=False,
                        select=False,
                    )
                elif task == "regression":
                    m = MyRSplineRegressor(
                        random_state=1337,
                        maxk=10,
                        m=2,
                        gamma=1,
                        model_to_use="gam",
                        # model_to_use=self.arg_dict["model_to_use"],
                        spline_type="ts",
                        discrete=False,
                        select=False,
                    )

                # fit the model
                m.fit(X, y)
                model_is_fitted = True
            except:
                message = f"For this seed: {seed} no fit was found."
                print(f"\033[91m{message}\033[0m")
                seed = seed + 10
                print(f"\033[0;32m{f'New seed is: {seed}!'}\033[0m")

        m.create_df_from_R_model(X)
        df = m.get_GAM_plot_dataframe()

        feat_row = df.loc[df["feat_name"] == feature_name]
        Xs.append(feat_row.x.values[0])
        Ys.append(feat_row.y.values[0])

    min_length = min(len(arr) for arr in Ys)
    Ys = [arr[:min_length] for arr in Ys]
    Xs = [arr[:min_length] for arr in Xs]

    Ys_lower_bound = np.mean(Ys, axis=0) - np.std(Ys, axis=0)
    Ys_upper_bound = np.mean(Ys, axis=0) + np.std(Ys, axis=0)
    X = np.mean(Xs, axis=0)
    Y = np.mean(Ys, axis=0)

    make_plot(
        X,
        Y,
        feature_name,
        model_name,
        dataset.name,
        scaler_dict,
        y_scaler,
        X_original,
        Ys_lower_bound,
        Ys_upper_bound,
    )


random_state = 1

dataset = Dataset("crimes", "")
feature_name = "racepctblack"

task = dataset.problem


# test if directory "plots" exists, if not create it
if not os.path.isdir("plots"):
    os.mkdir("plots")
if not os.path.isdir(f"plots/{dataset.name}"):
    os.mkdir(f"plots/{dataset.name}")


# EBM(dataset, feature_name)
# PYGAM(dataset, feature_name)
# Gaminet(dataset, feature_name)
Gaminet_complete(dataset, feature_name)
# LR(dataset, feature_name)
# I_GANN(dataset, feature_name)
# MGCV(dataset, feature_name)
