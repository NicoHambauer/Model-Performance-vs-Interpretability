import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# %
from load_datasets import Dataset

model_name = 'CATBOOST'

data = Dataset('adult', model_name)
X = data.X
y = data.y

X = pd.get_dummies(X, drop_first=True)
# drop first to avoid multicollinearity

# %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# reindex all the dataframes
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# %

hpo_grid = pd.read_json('default_hyperparams.json')
arg_dict = hpo_grid[model_name]
# usually we need to permute and iterate over all combinations of the hpo grid, here we simply choose one configuration

# arg_dict = {
#     "boost_rate": 0.1,
#     "elm_scale": 1,
#     "interactions": 5
# }
# %
from model import Model

model = Model(model_name, data.task, arg_dict, num_cols=data.numerical_cols, cat_cols=data.categorical_cols)

model.fit(X_train, y_train)

pred = model.predict(X_test)
# map predictions back to 0 and 1 instead of -1 and 1
# pred = np.where(pred == -1, 0, 1)
# from baseline.nam.utils import plot_nams
# %

#print all the required regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('MSE: ', mean_squared_error(y_test, pred))
print('MAE: ', mean_absolute_error(y_test, pred))
print('RMSE: ', mean_squared_error(y_test, pred, squared=False))
print('R2: ', r2_score(y_test, pred))

# from sklearn.metrics import classification_report
#
# cl_report = pd.DataFrame(classification_report(y_true=y_test, y_pred=pred, output_dict=True)).T
# print(cl_report)
#
# # print auroc
# from sklearn.metrics import roc_auc_score
#
# print('AUROC: ', roc_auc_score(y_test, model.predict_proba(X_test)))

# from baseline.nam.utils import plot_nams
# model.nam_to_cpu()
# fig = plot_nams(model.model.model, model.nam_dataset, num_cols= 2)