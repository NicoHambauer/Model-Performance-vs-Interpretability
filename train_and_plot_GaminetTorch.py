import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, Ridge
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from interpret import show
from pygam import LogisticGAM, LinearGAM
from gaminet import GAMINetRegressor, GAMINetClassifier
from baseline.exnn.exnn.exnn import ExNN
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from load_datasets import Dataset
import tensorflow as tf
import torch

import pytorch_lightning as pl
from baseline.nam.config import defaults
from baseline.nam.data import NAMDataset
from baseline.nam.models import NAM, get_num_units
from baseline.nam.trainer import LitNAM
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from typing import Dict
from dataclasses import dataclass, field

random_state = 1
task = 'classification'  # regression or classification

dataset = Dataset("adult")

X = dataset.X
y = dataset.y
X, y = shuffle(X, y, random_state=random_state)

transformers = [
    ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore', drop='if_binary'), dataset.categorical_cols),
    ('num', FunctionTransformer(), dataset.numerical_cols)
]
ct = ColumnTransformer(transformers=transformers, remainder='drop')
ct.fit(X)
X_original = X
X = ct.transform(X)

cat_cols = ct.named_transformers_['ohe'].get_feature_names_out(dataset.categorical_cols) if len(dataset.categorical_cols) > 0 else []
X = pd.DataFrame(X, columns=np.concatenate((cat_cols, dataset.numerical_cols)))

scaler_dict = {}
for c in dataset.numerical_cols:
    # sx = MinMaxScaler((0, 1))
    # sx.fit([[0], [1]])
    # X[c] = sx.transform(X[c].values.reshape(-1, 1))
    sx = StandardScaler()
    X[c] = sx.fit_transform(X[c].values.reshape(-1, 1))
    # scaler = RobustScaler()
    # X[c] = scaler.fit_transform(X[c].values.reshape(-1, 1))
    scaler_dict[c] = sx

# test if directory "plots" exists, if not create it
if not os.path.isdir('plots'):
    os.mkdir('plots')
@dataclass
class PlotData:
    _plot_dict: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def add_datapoint(self, column: str, label: str, value: float):
        if column not in self._plot_dict:
            self._add_column(column)
        self._plot_dict[column][label] = value

    def _add_column(self, name: str):
        self._plot_dict[name] = {}

    @property
    def entries(self):
        return self._plot_dict


def feature_importance_visualize(data_dict_global, folder="./results/", name="demo", save_png=False, save_eps=False):
    all_ir = []
    all_names = []
    for key, item in data_dict_global.items():
        if item["importance"] > 0:
            all_ir.append(item["importance"])
            all_names.append(key)

    max_ids = len(all_names)
    if max_ids > 0:
        fig = plt.figure(figsize=(0.4 + 0.6 * max_ids, 4))
        ax = plt.axes()
        ax.bar(np.arange(len(all_ir)), [ir for ir, _ in sorted(zip(all_ir, all_names))][::-1], color="grey")
        ax.set_xticks(np.arange(len(all_ir)))
        ax.set_xticklabels([name for _, name in sorted(zip(all_ir, all_names))][::-1], rotation=90)
        plt.xlabel("Feature Name", fontsize=12)
        plt.ylim(0, np.max(all_ir) + 0.05)
        plt.xlim(-1, len(all_names))
        plt.title("Feature Importance")

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

def make_plot(x, mean, upper_bounds, lower_bounds, feature_name, model_name, dataset_name, scale_back=True):
    x = np.array(x)
    if feature_name in dataset.numerical_cols and scale_back:
        x = scaler_dict[feature_name].inverse_transform(x.reshape(-1, 1)).squeeze()

    fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [0.8, 0.2]})
    ax1.plot(x, mean, color='black')

    if feature_name in dataset.numerical_cols:
        bins_values, _, _ = ax2.hist(X_original[feature_name], bins=10, color='grey') # X_original[feature_name] correct?
    else:
        bins_values, _, _ = ax2.hist(X[feature_name], bins=10, color='grey')


    ax1.set_title(f'Feature:{feature_name}')
    ax1.set_xlabel(f'Feature value')
    ax1.set_ylabel('Feature effect on model output')
    ax2.set_xlabel("Distribution")
    ax2.set_xticks([])
    ax2.set_yticks([0, max(bins_values)])
    fig.tight_layout()
    plt.savefig(f'plots/{model_name}_{dataset_name}_shape_{feature_name}.pdf')
    plt.show()


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


def make_plot_ebm(data_dict, feature_name, model_name, dataset_name, num_epochs='', debug=False):
    x_vals = data_dict["names"].copy()
    y_vals = data_dict["scores"].copy()

    # This is important since you do not plot plt.stairs with len(edges) == len(vals) + 1, which will have a drop to zero at the end
    y_vals = np.r_[y_vals, y_vals[np.newaxis, -1]]

    # This is the code interpretml also uses: https://github.com/interpretml/interpret/blob/2327384678bd365b2c22e014f8591e6ea656263a/python/interpret-core/interpret/visual/plot.py#L115

    # main_line = go.Scatter(
    #     x=x_vals,
    #     y=y_vals,
    #     name="Main",
    #     mode="lines",
    #     line=dict(color="rgb(31, 119, 180)", shape="hv"),
    #     fillcolor="rgba(68, 68, 68, 0.15)",
    #     fill="none",
    # )
    #
    # main_fig = go.Figure(data=[main_line])
    # main_fig.show()
    # main_fig.write_image(f'plots/{model_name}_{dataset_name}_shape_{feature_name}_{num_epochs}epochs.pdf')

    # This is my custom code used for plotting

    x = np.array(x_vals)
    if debug:
        print("Num cols:", dataset.numerical_cols)
    if feature_name in dataset.numerical_cols:
        if debug:
            print("Feature to scale back:", feature_name)
        x = scaler_dict[feature_name].inverse_transform(x.reshape(-1, 1)).squeeze()
    else:
        if debug:
            print("Feature not to scale back:", feature_name)

    fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [0.8, 0.2]})
    ax1.step(x, y_vals, where="post", color='black')
    bins_values, _, _ = ax2.hist(X_original[feature_name], bins=10, color='grey')

    #ax1.legend(loc='best')
    ax1.set_title(f'Feature:{feature_name}')
    ax1.set_xlabel(f'Feature value')
    ax1.set_ylabel('Feature effect on model output')
    ax2.set_xlabel("Distribution")
    ax2.set_xticks([])
    ax2.set_yticks([0, max(bins_values)])
    fig.tight_layout()

    #plt.xlabel(f'Feature value')
    #plt.ylabel('Feature effect on model output')
    #plt.title(f'Feature:{feature_name}')

    plt.savefig(f'plots/{model_name}_{dataset_name}_shape_{feature_name}_{num_epochs}epochs.pdf')
    plt.show()


def make_plot_interaction(left_names, right_names, scores, feature_name_left, feature_name_right, model_name,
                          dataset_name, scale_back=True):
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
        left_names = scaler_dict[feature_name_left].inverse_transform(left_names.reshape(-1, 1)).squeeze()
    if feature_name_right in dataset.numerical_cols and scale_back:
        right_names = scaler_dict[feature_name_right].inverse_transform(right_names.reshape(-1, 1)).squeeze()

    if "_" in feature_name_right:
        right_names = right_names.astype('str')

    if "_" in feature_name_left:
        left_names = left_names.astype('str')

    fig, ax = plt.subplots()
    im = ax.pcolormesh(left_names, right_names, scores, shading='auto')
    fig.colorbar(im, ax=ax)
    plt.xlabel(feature_name_left)
    plt.ylabel(feature_name_right)
    plt.title(f'Feature: {feature_name_left.replace("?", "missing")} x {feature_name_right.replace("?", "missing")}')
    plt.savefig(
        f'plots/{model_name}_{dataset_name}_interact_{feature_name_left.replace("?", "missing")} x {feature_name_right.replace("?", "missing")}.pdf')
    plt.show()


def make_plot_interaction_continous_x_cat_ebm(left_names, right_names, scores, feature_name_left, feature_name_right,
                                              model_name,
                                              dataset_name):
    """

    :param left_names:
    :param right_names:
    :param scores:
    :param feature_name_left:
    :param feature_name_right:
    :param model_name:
    :param dataset_name:
    :return:
    """
    left_names = np.array(left_names)
    # print(right_names)
    if feature_name_left in dataset.numerical_cols:
        left_names = scaler_dict[feature_name_left].inverse_transform(left_names.reshape(-1, 1)).squeeze()
    right_names = np.array(right_names)
    if feature_name_right in dataset.numerical_cols:
        right_names = scaler_dict[feature_name_right].inverse_transform(right_names.reshape(-1, 1)).squeeze()

    scores = np.r_[scores, scores[np.newaxis, -1]]
    scores = np.transpose(scores)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(left_names, right_names, scores, shading='auto')
    fig.colorbar(im, ax=ax)
    plt.xlabel(feature_name_left)
    plt.ylabel(feature_name_right)
    plt.savefig(
        f'plots/{model_name}_{dataset_name}_interact_{feature_name_left.replace("?", "")} x {feature_name_right.replace("?", "")}.pdf')
    plt.show()


def make_plot_interaction_onehot_line_gaminet(categorical_input, continuous_input, output,
                                              categorical_name,
                                              continuous_name,
                                              model_name,
                                              dataset_name):
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
        continuous_input = continuous_input.astype('str') #since we want categorical values and not a range
    bin_vals = output.transpose(-1, 0)
    bin_vals = np.ascontiguousarray(bin_vals)

    bin_vals_y_0 = bin_vals[0]
    bin_vals_y_1 = bin_vals[1]

    plt.subplots()
    plt.plot(continuous_input, bin_vals_y_0, color='black',
             label=f"{categorical_name} = {int(categorical_input[0])}")  # y_vals[0]
    plt.plot(continuous_input, bin_vals_y_1, color='grey',
             label=f"{categorical_name} = {int(categorical_input[1])}")  # y_vals[1]
    plt.legend(loc='best')
    # plt.fill_between(x, lower_bounds, mean, color='gray')
    # plt.fill_between(x, mean, upper_bounds, color='gray')
    plt.xlabel(f'Feature value')
    plt.ylabel('Feature effect on model output')
    plt.title(f'Feature:{categorical_name} x {continuous_name}')
    plt.savefig(f'plots/{model_name}_{dataset_name}_interact_{categorical_name} x {continuous_name}.pdf')
    plt.show()


def make_plot_interaction_onehot_line_ebm(data_dict,
                                          categorical_name,
                                          continuous_name,
                                          model_name,
                                          dataset_name):
    x_positions = np.array(data_dict["right_names"])
    categories = np.array(data_dict["left_names"])
    bin_vals = data_dict["scores"]
    bin_vals = np.ascontiguousarray(bin_vals)

    bin_vals_y_0 = bin_vals[0]
    bin_vals_y_1 = bin_vals[1]

    bin_vals_y_0 = np.r_[bin_vals_y_0, bin_vals_y_0[np.newaxis, -1]]
    bin_vals_y_1 = np.r_[bin_vals_y_1, bin_vals_y_1[np.newaxis, -1]]

    if continuous_name in dataset.numerical_cols:
        x_positions = scaler_dict[continuous_name].inverse_transform(x_positions.reshape(-1, 1)).squeeze()
        # inverse_transform x_positions with the one hot encoder


    fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [0.8, 0.2]})
    ax1.step(x_positions, bin_vals_y_0, where="post", color='black',
             label=f"{categorical_name} = {int(float(categories[0]))}")  # y_vals[0]
    ax1.step(x_positions, bin_vals_y_1, where="post", color='grey',
             label=f"{categorical_name} = {int(float(categories[1]))}") # y_vals[1]
    bins_values, _, _ = ax2.hist(X_original[continuous_name], bins=10, color='grey')

    ax1.legend(loc='best')
    ax1.set_title(f'Feature:{categorical_name} x {continuous_name}')
    ax1.set_xlabel(f'Feature value: {continuous_name}')
    ax1.set_ylabel('Feature effect on model output')
    ax2.set_xlabel("Distribution")
    ax2.set_xticks([])
    ax2.set_yticks([0, max(bins_values)])
    fig.tight_layout()

    #fig.title(f'Feature:{categorical_name} x {continuous_name}')
    plt.savefig(f'plots/{model_name}_{dataset_name}_interact_{categorical_name} x {continuous_name}.pdf')
    plt.show()


def plot_pairwise_heatmap(data_dict,
                          x_name,
                          y_name,
                          model_name,
                          dataset_name,
                          title="",
                          xtitle="",
                          ytitle=""):
    if data_dict.get("scores", None) is None:  # pragma: no cover
        return None

    bin_labels_left_x = np.array(data_dict["left_names"])
    bin_labels_right_y = np.array(data_dict["right_names"])
    bin_vals = data_dict["scores"]
    bin_vals = np.ascontiguousarray(np.transpose(bin_vals, (1, 0)))

    if x_name in dataset.numerical_cols:
        bin_labels_left_x = scaler_dict[x_name].inverse_transform(bin_labels_left_x.reshape(-1, 1)).squeeze()

    heatmap = go.Heatmap(z=bin_vals, x=bin_labels_left_x, y=bin_labels_right_y, colorscale="Viridis")

    # matplotlib Colormesh -- Only working with
    fig, ax = plt.subplots()
    # import plotly.express as px
    im = ax.pcolormesh(bin_labels_left_x, bin_labels_right_y, bin_vals, shading='auto')
    fig.colorbar(im, ax=ax)
    plt.title(f'Feature: {x_name.replace("?", "")} x {y_name.replace("?", "")}')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig(f'plots/{model_name}_{dataset_name}_interact_{x_name.replace("?", "")} x {y_name.replace("?", "")}.pdf')
    plt.show()

    if data_dict.get("scores_range", None) is not None:
        heatmap["zmin"] = data_dict["scores_range"][0]
        heatmap["zmax"] = data_dict["scores_range"][1]

def make_one_hot_plot(class_zero, class_one, feature_name, model_name, dataset_name):
    # This snippet is for binary plots
    original_feature_name = feature_name.split('_')[0]
    if X_original[original_feature_name].value_counts().size == 2:
        category_0 = X_original[original_feature_name].values.categories[0]
        category_1 = X_original[original_feature_name].values.categories[1]
        categories = [category_0, category_1]
        plt.bar([0, 1], [class_zero, class_one], color='gray', tick_label=[f'{categories[0]}', f'{categories[1]} '])

    # This snippet is for all plots but binary categories
    # TODO: Binary-Class One Hot Plot für Multi-Class Plots rausnehmen
    else:
        plt.bar([0, 1], [class_zero, class_one], color='gray',
                tick_label=[f'{feature_name} = 0', f'{feature_name} = 1'])

    plt.ylabel('Feature effect on model output')
    plt.title(f'Feature:{feature_name.split("_")[0]}')
    plt.savefig(f'plots/{model_name}_{dataset_name}_onehot_{str(feature_name).replace("?", "")}.pdf')
    plt.show()


def make_one_hot_multi_plot(plot_data: PlotData, model_name, dataset_name):
    for feature_name in plot_data.entries:
        position_list = range(0, len(plot_data.entries[feature_name]))
        y_list = list(plot_data.entries[feature_name].values())
        x_list = list(plot_data.entries[feature_name].keys())
        plt.bar(position_list, y_list, color='gray', tick_label=x_list)
        plt.xticks(rotation=90)
        plt.ylabel('Feature effect on model output')
        plt.title(f'Feature:{feature_name}')
        #print(feature_name)
        plt.savefig(f'plots/{model_name}_{dataset_name}_multi_onehot_{str(feature_name).replace("?", "")}.pdf',
                    bbox_inches="tight")
        plt.show()


# %%

def EBM_show(X, y):
    m4 = ExplainableBoostingRegressor(interactions=10, max_bins=256)
    m4.fit(X, y)
    ebm_global = m4.explain_global()
    show(ebm_global)


def EBM(X, y, dataset_name, model_name='EBM'):
    if task == "classification":
        ebm = ExplainableBoostingClassifier(interactions=20, max_bins=256)
    else:
        ebm = ExplainableBoostingRegressor(interactions=20, max_bins=256)
    ebm.fit(X, y)
    ebm_global = ebm.explain_global()

    plot_data = PlotData()
    for i, _ in enumerate(ebm_global.data()['names']):  # shape_name besserer Name?
        data_names = ebm_global.data()
        print(data_names['names'][i])
        feature_name = data_names['names'][i]
        shape_data = ebm_global.data(i)

        print(feature_name)

        if shape_data['type'] == 'interaction':
            x_name, y_name = feature_name.split(' x ')

            if len(shape_data['left_names']) == 2 and len(shape_data['right_names']) > 2:
                # plot_pairwise_heatmap(shape_data, x_name, y_name, model_name, dataset_name)
                make_plot_interaction_onehot_line_ebm(shape_data, x_name, y_name, model_name, dataset_name)
            elif len(shape_data['left_names']) > 2 and len(shape_data['right_names']) == 2:
                raise ValueError('Left Names contains continue values. We thought that should not happen. Fix it.')
            elif len(shape_data['left_names']) == 2 and len(shape_data['right_names']) == 2:
                # TODO: cat x cat verify
                plot_pairwise_heatmap(shape_data, x_name, y_name, model_name, dataset_name)
            else:
                make_plot_interaction(shape_data['left_names'], shape_data['right_names'],
                np.transpose(shape_data['scores']),
                x_name, y_name, model_name, dataset_name)

            continue

        elif (shape_data['type'] == 'univariate'):

            # if feature only has two categories, make single one hot plot
            original_feature_name = feature_name.split('_')[0]
            if X_original[original_feature_name].value_counts().size == 2:
                make_one_hot_plot(shape_data['scores'][0], shape_data['scores'][1], feature_name, model_name,
                                  dataset_name)

            # else, if feature has more than two categories, make multi one hot plot
            elif feature_name.split('_')[0] not in dataset.numerical_cols:
                column_name = feature_name.split('_')[0]
                class_name = feature_name.split('_')[1]
                class_score = shape_data['scores'][1]

                plot_data.add_datapoint(column_name, class_name, class_score)

            else:
                make_plot_ebm(shape_data, feature_name, model_name, dataset_name)
                # plot_continuous_bar(shape_data, feature_name, model_name, dataset_name)

    # this function call uses the collected plot_data to plot all multi one hot plots
    make_one_hot_multi_plot(plot_data, model_name, dataset_name)

    feat_for_vis = dict()
    for i, n in enumerate(ebm_global.data()['names']):
        feat_for_vis[n] = {'importance': ebm_global.data()['scores'][i]}
    feature_importance_visualize(feat_for_vis, save_png=True, folder='.', name='ebm_feat_imp')

    #ebm_local = ebm.explain_local(X[:5], y[:5], name='EBM_local')
    #show(ebm_local)
    #import time
    #time.sleep(1000)


# def modify_interaction_ranges(ebm_global, min_heatmap_val, max_heatmap_val):
#     for data_dict in ebm_global._internal_obj['specific']:
#         if data_dict['type'] == 'interaction':
#             data_dict['scores_range'] = (min_heatmap_val, max_heatmap_val)

def PYGAM(X, y, dataset_name, model_name='PYGAM'):
    # TODO: Integrate terms as parameters on model initialization
    if task == "classification":
        PYGAM = LogisticGAM().fit(X, y)
    elif task == "regression":
        PYGAM = LinearGAM().fit(X, y)

    plot_data = PlotData()
    for i, term in enumerate(PYGAM.terms):
        if term.isintercept:
            continue
        XX = PYGAM.generate_X_grid(term=i)
        pdep, confi = PYGAM.partial_dependence(term=i, X=XX, width=0.95)
        # make_plot(XX[:,i].squeeze(), pdep, confi[:,0], confi[:,1], X.columns[i])

        original_feature_name = X[X.columns[i]].name.split("_")[0]
        if (X_original[original_feature_name].value_counts().size > 2) and (original_feature_name in dataset.categorical_cols):
            column_name = original_feature_name
            class_name = X[X.columns[i]].name.split("_")[1]
            class_score = pdep[-1]

            plot_data.add_datapoint(column_name, class_name, class_score)

        if len(X[X.columns[i]].unique()) == 2:
            make_one_hot_plot(pdep[0], pdep[-1], X.columns[i], model_name, dataset_name)
        else:
            make_plot(XX[:, i].squeeze(), pdep, pdep, pdep, X.columns[i], model_name, dataset_name)

    make_one_hot_multi_plot(plot_data, model_name, dataset_name)


def Gaminet(X, y, dataset_name, model_name='Gaminet'):
    x_types = {}
    for i in range(len(X.columns)):
        if "_" in X.columns[i]:
            x_types[X.columns[i]] = {'type': 'categorical', "values": [0, 1]}
        else:
            x_types[X.columns[i]] = {'type': 'continuous'}

    # meta_info = {X.columns[i]: {'type': 'continuous'} for i in range(len(X.columns))}
    meta_info = x_types
    meta_info.update({'Y': {'type': 'target'}})

    # from sklearn.preprocessing import FunctionTransformer
    identity = FunctionTransformer()

    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == 'target':
            continue
        # sx = MinMaxScaler((0, 1))
        # sx.fit([[0], [1]])
        # print(scaler_dict.keys())
        # print(X.columns)
        if key in scaler_dict and item['type'] != 'categorical':
            meta_info[key]['scaler'] = scaler_dict[key]
        else:
            meta_info[key]['scaler'] = identity
    #X = torch.tensor(X.values, dtype=torch.float32)
    #y = torch.tensor(y.values, dtype=torch.float32)
    if task == "classification": # 190 it/ s Maineffect, 65 it/s Interaction
        model_to_run = GAMINetClassifier(meta_info=meta_info, interact_num=10,
                               batch_size=1024, activation_func='ReLU',
                               max_epochs=(5000, 5000, 500),
                               learning_rates=[0.0001, 0.0001, 0.0001], early_stop_thres=[50, 50, 50],
                               heredity=False, loss_threshold=0.01, reg_clarity=1,  # clarity 1.0
                               mono_increasing_list=[], mono_decreasing_list=[],  # the indices list of features
                               verbose=True, val_ratio=0.2, random_state=random_state)
        print(np.array(y).shape)
        model_to_run.fit(np.array(X), np.array(y).reshape(-1, 1))

    elif task == "regression":
        model_to_run = GAMINetRegressor(meta_info=meta_info, interact_num=10,
                               batch_size=1024, activation_func='ReLU',
                               main_effect_epochs=5000, interaction_epochs=5000, tuning_epochs=500,
                               lr_bp=[0.0001, 0.0001, 0.0001], early_stop_thres=[50, 50, 50],
                               heredity=True, loss_threshold=0.01, reg_clarity=1,
                               mono_increasing_list=[], mono_decreasing_list=[],  # the indices list of features
                               verbose=True, val_ratio=0.2, random_state=random_state)
        model_to_run.fit(np.array(X), np.array(y))

    data_dict = model_to_run.global_explain(main_grid_size=256)

    Xnames2Featurenames = dict(zip([X.columns[i] for i in range(X.shape[1])], X.columns))
    print(Xnames2Featurenames)

    plot_data = PlotData()
    for effect in data_dict.keys():
        if data_dict[effect]['type'] == 'pairwise':
            x_name, y_name = effect.split(' vs. ')
            # feature_name_left = feature_name_left.replace(' ', '')
            # feature_name_right = feature_name_right.replace(' ', '')

            if len(data_dict[effect]['input1']) == 2 and len(
                    data_dict[effect]['input2']) > 2:  # if there is continous x categirical
                make_plot_interaction_onehot_line_gaminet(data_dict[effect]['input1'], data_dict[effect]['input2'],
                                                          data_dict[effect]['outputs'], x_name, y_name, model_name,
                                                          dataset_name)
            else:
                make_plot_interaction(data_dict[effect]['input1'], data_dict[effect]['input2'], data_dict[effect]['outputs'],
                                      x_name,
                                      y_name,
                                      model_name, dataset_name, scale_back=False)

        elif data_dict[effect]['type'] == 'continuous':
            make_plot(data_dict[effect]['inputs'], data_dict[effect]['outputs'], data_dict[effect]['outputs'],
                      data_dict[effect]['outputs'], Xnames2Featurenames[effect], model_name, dataset_name, scale_back=False)

        elif data_dict[effect]['type'] == 'categorical':  ####len(X[Xnames2Featurenames[k]].unique()) == 2:
            make_one_hot_plot(data_dict[effect]['outputs'][0], data_dict[effect]['outputs'][-1],
                              Xnames2Featurenames[effect], model_name, dataset_name)

            column_name = Xnames2Featurenames[effect].split("_")[0]
            class_name = Xnames2Featurenames[effect].split("_")[1]
            class_score = data_dict[effect]['outputs'][-1]
            plot_data.add_datapoint(column_name, class_name, class_score)

        else:
            continue

        make_one_hot_multi_plot(plot_data, model_name, dataset_name)

    feat_for_vis = dict()
    for i, effect in enumerate(data_dict.keys()):
        if 'vs.' in effect:
            feature_name_left, feature_name_right = effect.split(' vs. ')

            feat_for_vis[f'{feature_name_left}\nvs.\n{feature_name_right}'] = {'importance': data_dict[effect]['importance']}
        else:
            feat_for_vis[Xnames2Featurenames[effect]] = {'importance': data_dict[effect]['importance']}

    feature_importance_visualize(feat_for_vis, save_png=True, folder='.', name='gaminet_feat_imp')


def EXNN(X, y, dataset_name, model_name='ExNN'):
    meta_info = {"X" + str(i + 1): {'type': 'continuous'} for i in range(len(X.columns))}
    meta_info.update({'Y': {'type': 'target'}})

    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == 'target':
            continue
        sx = MinMaxScaler((0, 1))
        sx.fit([[0], [1]])
        meta_info[key]['scaler'] = sx

    X_arr = np.array(X)
    y_arr = np.array(y)

    if task == "classification":
        model_to_run = ExNN(meta_info=meta_info, subnet_num=10, subnet_arch=[10, 6], task_type="Classification",
                            activation_func=tf.tanh, batch_size=min(1000, int(X.shape[0] * 0.2)),
                            training_epochs=10000,  # default 10000
                            lr_bp=0.001, lr_cl=0.1, beta_threshold=0.05, tuning_epochs=100, l1_proj=0.0001,

                            l1_subnet=0.00316,
                            l2_smooth=10 ** (-6), verbose=True, val_ratio=0.2, early_stop_thres=500)
        model_to_run.fit(X_arr, y_arr)

        model_to_run.visualize(save_png=True, folder='plots/', name=f'{model_name}_{dataset_name}_shape')

    elif task == "regression":
        model_to_run = ExNN(meta_info=meta_info, subnet_num=10, subnet_arch=[10, 6], task_type="Regression",
                            activation_func=tf.tanh, batch_size=min(1000, int(X.shape[0] * 0.2)),
                            training_epochs=10000,  # default
                            lr_bp=0.001, lr_cl=0.1, beta_threshold=0.05, tuning_epochs=100, l1_proj=0.0001,
                            l1_subnet=0.00316,
                            l2_smooth=10 ** (-6), verbose=True, val_ratio=0.2, early_stop_thres=500)
        model_to_run.fit(X_arr, y_arr)

        model_to_run.visualize(save_png=True, folder='plots/', name=f'{model_name}_{dataset_name}_shape')


def LR(X, y, dataset_name, model_name='LR'):
    # TODO: Implement if classification / regression
    m = Ridge()
    # if task == 'regression':
    # else:
    #     m = LogisticRegression()
    m.fit(X, y)
    import seaborn as sns
    # plot = sns.distplot(m.coef_)
    word_to_coef = dict(zip(m.feature_names_in_, m.coef_.squeeze()))
    dict(sorted(word_to_coef.items(), key=lambda item: item[1]))
    word_to_coef_df = pd.DataFrame.from_dict(word_to_coef, orient='index')
    print(word_to_coef_df)

    for i, feature_name in enumerate(X.columns):
        inp = torch.linspace(X[feature_name].min(), X[feature_name].max(), 1000)
        outp = word_to_coef[feature_name] * inp
        # outp = nam_model.feature_nns[i](inp).detach().numpy().squeeze()
        # if len(X[feature_name].unique()) == 2:
        #     make_one_hot_plot(outp[0], outp[-1], feature_name, model_name, dataset_name)
        # else:
        make_plot(inp, outp, outp, outp, feature_name, model_name, dataset_name)


def NAM_runner(X, y, dataset_name, model_name='NAM'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = defaults()
    config.early_stopping_patience == 60
    config.num_epochs = 1000
    config.num_basis_functions = 1000
    config.decay_rate = 0.995
    config.activation = 'relu'  # 'exu'
    config.dropout = 0.5
    config.units_multiplier = 2
    config.optimizer = 'adam'
    config.output_regularization = 0.0
    config.feature_dropout = 0.0
    config.logdir = "../models/NAM"
    config.l2_regularization = 0.0
    config.batch_size = 1024
    config.lr = 0.01  # 0.0003

    if task == "classification":
        config.regression = False
    elif task == "cegression":
        config.regression = True

    X['target'] = y
    dataset = X
    dataset = NAMDataset(config, data_path=dataset, features_columns=dataset.columns[:-1],
                         targets_column=dataset.columns[-1])

    train_idx = np.arange(len(X)).tolist()
    test_idx = np.arange(len(X)).tolist()
    dataset.setup_dataloaders_custom(train_idx[0:int(len(train_idx) * 0.9)],
                                     train_idx[int(len(train_idx) * 0.9):],
                                     test_idx)
    dataloaders = dataset.train_dataloaders()

    nam_model = NAM(
        config=config,
        name="NAM",
        num_inputs=len(dataset[0][0]),
        num_units=get_num_units(config, dataset.features),
    )
    nam_model = nam_model.to(device)

    for fold, (trainloader, valloader) in enumerate(dataloaders):
        litmodel = LitNAM(config, nam_model)
        litmodel = litmodel.to(device)
        tb_logger = TensorBoardLogger(save_dir="../models/NAM_Plot",
                                      name=f'{nam_model.name}')

        checkpoint_callback = ModelCheckpoint(filename=tb_logger.log_dir + "/{epoch:02d}-{val_loss:.4f}",
                                              monitor='val_loss',
                                              save_top_k=config.save_top_k,
                                              mode='min')

        trainer = pl.Trainer(
            max_epochs=config.num_epochs,
            callbacks=[checkpoint_callback])

        trainer.fit(litmodel, train_dataloader=trainloader, val_dataloaders=valloader)

    for i, feature_name in enumerate(X.drop('target', axis=1).columns):
        inp = torch.linspace(X[feature_name].min(), X[feature_name].max(), 1000)
        outp = nam_model.feature_nns[i](inp).detach().numpy().squeeze()
        if len(X[feature_name].unique()) == 2:
            make_one_hot_plot(outp[0], outp[-1], feature_name, model_name, dataset_name, config.num_epochs)
        else:
            make_plot(inp, outp, outp, outp, feature_name, model_name, dataset_name, config.num_epochs)


# EBM_show(X, y) # for EBM_Show copy paste this script into a jupyter notebook and only run the EBM_Show dashboard
EBM(X, y, dataset.name)
#PYGAM(X, y, dataset.name)
Gaminet(X, y, dataset.name)
#EXNN(X, y, dataset.name)
#LR(X, y, dataset.name)



