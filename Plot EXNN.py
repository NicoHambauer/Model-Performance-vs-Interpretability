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

dataset = Dataset("crimes")
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


# arg_dict_exnn = {
# "subnet_num": [5],
#     "l1_proj": [0.001],
#     "l1_subnet": [0.001]
# }

# arg_dict_exnn_tuned = {
#     'subnet_num': 5,
#     'l1_proj': 0.01,
#     'l1_subnet': 0.01
# }

arg_dict_exnn_gam_version = {
    'subnet_num': 5,
    'l1_proj': 0.01,
    'l1_subnet': 0.01,
    'blabla': 0.01
}

model_name = 'EXNN'

m = Model(model_name, dataset.problem, arg_dict_exnn_gam_version, dataset.numerical_cols, dataset.categorical_cols)
m.fit(X, y)

m.model.visualize(save_png=True, folder=f'plots/{dataset.name}/', name=f'{model_name}_shape')