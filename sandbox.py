import json

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# %
from load_datasets import Dataset

model_name = 'CATBOOST'

data = Dataset('adult', model_name)
X = data.X
y = data.y

problem = data.problem
numerical_cols = data.numerical_cols
categorical_cols = data.categorical_cols

# # split val the data into train and val
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1337)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', "passthrough", categorical_cols) #OneHotEncoder(sparse=False, handle_unknown='ignore', drop='if_binary') if model_name != "CATBOOST" else
    ])

# Apply preprocessing to the dataset
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

X_train = pd.DataFrame(X_train, columns=preprocessor.get_feature_names_out())
X_test = pd.DataFrame(X_test, columns=preprocessor.get_feature_names_out())

# y_train = y_train.reset_index(drop=True)
# y_test = y_test.reset_index(drop=True)

if problem == "regression":
    y_scaler = StandardScaler()
    # scale the target out of sample for regression
    y_train = np.array(y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten())
    y_test = np.array(y_scaler.transform(y_test.values.reshape(-1, 1)).flatten())

# %

with open('default_hyperparams.json', "r") as read_file:
    hpo_grid = json.load(read_file)

arg_dict = hpo_grid[model_name]
# usually we need to permute and iterate over all combinations of the hpo grid, here we simply choose one configuration
# get the first value for each key and use it as the hyperparameter
arg_dict = {k: v[0] for k, v in arg_dict.items()}
# arg_dict['interactions'] = 0

# %%

all_transformed_feature_names = preprocessor.get_feature_names_out()

transformed_numerical_names = [name for name in all_transformed_feature_names if name.startswith('num__')]
transformed_categorical_names = [name for name in all_transformed_feature_names if name.startswith('cat__')]


# %%
from model import Model

model = Model(model_name, problem, arg_dict, num_cols=transformed_numerical_names, cat_cols=transformed_categorical_names)
# model = ExplainableBoostingClassifier(random_state=42)

model.fit(X_train, y_train)

pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
# for igann map predictions back to 0 and 1 instead of -1 and 1
# pred = np.where(pred == -1, 0, 1)

# print all the required regression metrics
if problem == "regression":
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    print('MSE: ', mean_squared_error(y_test, pred))
    print('MAE: ', mean_absolute_error(y_test, pred))
    print('RMSE: ', mean_squared_error(y_test, pred, squared=False))
    print('R2: ', r2_score(y_test, pred))

if problem == "classification":
    from sklearn.metrics import classification_report, roc_auc_score
    cl_report = pd.DataFrame(classification_report(y_true=y_test, y_pred=pred, output_dict=True)).T
    print(cl_report)
    print('AUROC: ', roc_auc_score(y_test, y_pred_proba[:,1]))