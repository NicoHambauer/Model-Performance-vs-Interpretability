# SPDX-FileCopyrightText: 2024 Nico Hambauer, Sven Kruschel
#
# SPDX-License-Identifier: MIT

import json
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.utils import shuffle
from load_datasets import Dataset
from model import Model

random_state = 1

dataset = Dataset("productivity")
task = dataset.problem  # regression or classification

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

cat_cols = ct.named_transformers_['ohe'].get_feature_names_out(dataset.categorical_cols) if len(
    dataset.categorical_cols) > 0 else []
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
if not os.path.isdir(f'plots/{dataset.name}'):
    os.mkdir(f'plots/{dataset.name}')


def make_plot(X, Y, feature_name, model_name, dataset_name, scale_back=True, scale_y=False, distribution_plot = True):

    X = np.array(X)
    Y = np.array(Y)

    # rescale numerical features
    if feature_name in dataset.numerical_cols and scale_back:
        X = scaler_dict[feature_name].inverse_transform(X.reshape(-1, 1)).squeeze()

    # create distribution plot
    if distribution_plot:
        fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [0.8, 0.2]})
        if feature_name in dataset.numerical_cols:
            bins_values, _, _ = ax2.hist(X_original[feature_name], bins=10, rwidth=0.9, color='grey')
        else:
            bins_values, _, _ = ax2.hist(X[feature_name], bins=10, rwidth=0.9, color='grey')
        ax2.set_xlabel("Distribution")
        ax2.set_xticks([])
        ax2.set_yticks([0, max(bins_values)])

    else:
        fig, ax1 = plt.subplots(nrows=1)

    fig.set_size_inches(5, 5)
    fig.set_dpi(100)

    ax1.plot(X, Y, color='black',alpha = 1)
    ax1.set_title(f'Feature:{feature_name}')
    ax1.set_xlabel(f'Feature value')
    ax1.set_ylabel('Feature effect on model output')
    fig.tight_layout()
    plt.savefig(f'plots2/{dataset_name}/{model_name}_shape_{feature_name}.png')
    plt.show()
    plt.close(fig)


# arg_dict_nam = {
#     "lr": 0.02082,
#     "num_learners": 1,
#     "dropout": 0.1,
#     "num_basis_functions": 64
# }

arg_dict_nam_tuned = {
    'lr': 0.01, 'num_learners': 1,
 'dropout': 0, 'num_basis_functions': 64
}

m = Model('NAM', dataset.problem, arg_dict_nam_tuned, dataset.numerical_cols, dataset.categorical_cols)
m.fit(X, y)

features = ['targeted productivity']

model_name = 'NAM'

for feature in features:
    # get index of feature in X
    i = X.columns.get_loc(feature)
    plot_dict = m.model.plot(i)
    make_plot(plot_dict['x'], plot_dict['y'], feature, model_name, dataset.name, scale_back=True)