import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from load_datasets import Dataset
import pandas as pd

regression_datasets = ['water', 'stroke', 'telco', 'fico', 'bank', 'adult',
                       'airline', 'college', 'weather', 'compas']

categorical_datasets = ['car', 'student', 'crimes', 'bike', 'housing',
                        'medical', 'crab', 'wine', 'diamond', 'productivity']

header = [['', '', 'No Preprocessing', 'No Preprocessing', 'No Preprocessing', 'Preprocessing', 'Preprocessing', 'Preprocessing', 'Encoded', 'Encoded'],
          ['Task', 'Dataset Name', 'Samples', 'Columns', 'Columns', 'Samples', 'Columns', 'Columns', 'Columns', 'Columns'],
          ['', '', '', 'num' ,'cat', '', 'num', 'cat', 'num', 'cat']]

original_dataset_metadata = {
    'water': {"num": 9, "cat": 0},
    'stroke': {"num": 4, "cat": 7},
    'telco': {"num": 3, "cat": 17},
    'fico': {"num": 21, "cat": 2},
    'bank': {"num": 10, "cat": 10},
    'adult': {"num": 6, "cat": 8},
    'airline': {"num": 20, "cat": 4},
    'college': {"num": 4, "cat": 6},
    'weather': {"num": 18, "cat": 5},
    'compas': {"num": 31, "cat": 22},
    'car': {"num": 14, "cat": 11},
    'student': {"num": 13, "cat": 17},
    'crimes': {"num": 125, "cat": 2},
    'bike': {"num": 11, "cat": 5},
    'housing': {"num": 8, "cat": 0},
    'medical': {"num": 3, "cat": 3},
    'crab': {"num": 7, "cat": 1},
    'wine': {"num": 11, "cat": 0},
    'diamond': {"num": 7, "cat": 3},
    'productivity': {"num": 10, "cat": 4}
}

dataset_metatable = {}
for dataset_name in regression_datasets + categorical_datasets:
    dummy_dataset = Dataset(dataset_name)

    X = dummy_dataset.X.copy()
    y = dummy_dataset.y.copy()
    transformers = [
        ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore', drop='if_binary'), dummy_dataset.categorical_cols),
        ('num', FunctionTransformer(), dummy_dataset.numerical_cols)
    ]
    ct = ColumnTransformer(transformers=transformers, remainder='drop')
    ct.fit(X)
    X_original = X
    X = ct.transform(X)

    cat_cols = ct.named_transformers_['ohe'].get_feature_names_out(dummy_dataset.categorical_cols) if len(
        dummy_dataset.categorical_cols) > 0 else []
    X = pd.DataFrame(X, columns=np.concatenate((cat_cols, dummy_dataset.numerical_cols)))

    dataset_metatable[dataset_name] = [dummy_dataset.problem,
                                       dataset_name,
                                       dummy_dataset.basic_dataset_metadata['n_samples'],
                                       original_dataset_metadata[dataset_name]["num"], # np.nan because we don't automatically know how many numerical columns there are
                                       original_dataset_metadata[dataset_name]["cat"], # np.nan because we don't automatically know how many categorical columns there are
                                       dummy_dataset.preprocessing_dataset_metadata['n_samples'],
                                       len(dummy_dataset.numerical_cols),
                                       len(dummy_dataset.categorical_cols),
                                       len(dummy_dataset.numerical_cols),
                                       len(cat_cols)]

#print(dataset_metatable)



df = pd.DataFrame.from_dict(dataset_metatable, orient='index', columns=header)

for row, index in df.iterrows():
    diff_preprocessing_samples = df.loc[row, ('Preprocessing', 'Samples')] - df.loc[row, ('No Preprocessing', 'Samples')]
    diff_preprocessing_columns_num = df.loc[row, ('Preprocessing', 'Columns', 'num')] - df.loc[row, ('No Preprocessing', 'Columns', 'num')]
    diff_preprocessing_columns_cat = df.loc[row, ('Preprocessing', 'Columns', 'cat')] - df.loc[row, ('No Preprocessing', 'Columns', 'cat')]
    diff_encoded_difference_cat = df.loc[row, ('Encoded', 'Columns', 'cat')] - df.loc[row, ('Preprocessing', 'Columns', 'cat')]

    df.loc[row, ('Preprocessing', 'Samples')] = str(df.loc[row, ('Preprocessing', 'Samples')]) + f' ({"+" if diff_preprocessing_samples > 0 else ""}{str(diff_preprocessing_samples)})'
    df.loc[row, ('Preprocessing', 'Columns', 'num')] = str(df.loc[row, ('Preprocessing', 'Columns', 'num')]) + f' ({"+" if diff_preprocessing_columns_num > 0 else ""}{str(diff_preprocessing_columns_num)})'
    df.loc[row, ('Preprocessing', 'Columns', 'cat')] = str(df.loc[row, ('Preprocessing', 'Columns', 'cat')]) + f' ({"+" if diff_preprocessing_columns_cat > 0 else ""}{str(diff_preprocessing_columns_cat)})'
    df.loc[row, ('Encoded', 'Columns', 'cat')] = str(df.loc[row, ('Encoded', 'Columns', 'cat')]) + f' ({"+" if diff_encoded_difference_cat > 0 else ""}{str(diff_encoded_difference_cat)})'

with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):
    print(df)
df.to_latex('./dataset_table.tex', index=False)
