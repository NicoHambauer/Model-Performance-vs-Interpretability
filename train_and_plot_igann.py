import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from sklearn.utils import shuffle

from igann import IGANN

from load_datasets import Dataset

dataset = Dataset("adult")

X = dataset.X
y = dataset.y
X, y = shuffle(X, y, random_state=0)


transformers = [
    ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore', drop='if_binary'), dataset.categorical_cols),
    ('num', FunctionTransformer(), dataset.numerical_cols)
]
ct = ColumnTransformer(transformers=transformers, remainder='drop')
ct.fit(X)
X_original = X
X = ct.transform(X)
cat_cols = ct.named_transformers_['ohe'].get_feature_names_out(dataset.categorical_cols)
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

# %%

igann = IGANN(task=dataset.problem, n_estimators=2000,
           # n_hid=n_hid,
           # elm_alpha=elm_alpha,
           # elm_scale=elm_scale,
           # init_reg=reg,
           early_stopping=50,
           interactions=10,
           # boost_rate=boost_rate,
           device="cuda",
           verbose=1)

# %%
igann.fit(X[:2000], y[:2000])
# m1.plot_single(plot_by_list=['racePctAsian', 'racePctWhite', 'racepctblack', 'racePctHisp'], scaler_dict=scaler_dict)

igann.plot_interactions_continous_x_onehot_bacs(scaler_dict=scaler_dict)
