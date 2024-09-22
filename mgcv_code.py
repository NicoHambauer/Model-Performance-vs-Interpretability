import os

# os.environ["R_HOME"] = r"C:\Program Files\R\R-4.3.3"
# os.environ["PATH"] = r"C:\Program Files\R\R-4.3.3\bin\x64" + ";" + os.environ["PATH"]

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
from vis_utils import vis_main_effects
import matplotlib.pyplot as plt
from rpy2.robjects import pandas2ri, packages
from arch import MyRSplineClassifier
import numpy as np

pandas2ri.activate()

utils = packages.importr("utils")
utils.chooseCRANmirror(ind=1)
stats = packages.importr("stats")
base = packages.importr("base")
mgcv_ = utils.install_packages("mgcv")
mgcv = packages.importr("mgcv")

breast = load_breast_cancer()
feature_names = list(breast.feature_names)
X, y = pd.DataFrame(breast.data, columns=feature_names), pd.Series(breast.target)
X.head()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

## bam + discrete can speed up a lot but sometimes would not work. Like the following would not work with this dataset:
# rspline = MyRSplineClassifier(maxk=10, discrete=True, select=False, model_to_use='bam')

## Maxk is also hard to tune. For this dataset if you tune it like 20, the package would give you a warning.
## And the result is wierd in this dataset with logodds (risk) value being around 10000
rspline = MyRSplineClassifier(
    spline_type="cr",
    maxk=10,
    discrete=False,
    select=False,
    model_to_use="gam",
)
rspline.fit(X_train, y_train)

preds_proba = rspline.predict_proba(X_test)
preds = [np.argmax(pred_proba) for pred_proba in preds_proba]

# To create the dataframe from this R model.
rspline.create_df_from_R_model(X_test)

fig, axes = vis_main_effects(
    {
        #    'XGB': xgb.get_GAM_plot_dataframe(),
        #    'EBM': ebm.get_GAM_plot_dataframe(),
        #    'Spline': spline.get_GAM_plot_dataframe(),
        #    'FLAM': flam.get_GAM_plot_dataframe(),
        "rspline": rspline.get_GAM_plot_dataframe(),  # the graph looks crazy
        #     'lr': lr.get_GAM_plot_dataframe(),
    }
)

plt.show()

# Parameter:
"""
Default Values mit unterschiedlichen Spline Types.
model: [bam, gam]

spline type: (pdf. 282)
"s(%s, bs='cs', k=%d)" --> cubic spline with penalization | 2 knots
"s(%s, bs='cr', k=%d)" --> cubic spline with penalization and shrinkage | 10 knots
"s(%s, bs='ts', k=%d)" --> thin plate spline with a modification of smoothing penalty
P-Spline: "bs"
D-Spline: "ds"
Thin Spline with Modification: "tp"


method: lassen wir erstmal weg
optimizer: lassen wir erstmal weg
k: default sind wohl 100

"""
