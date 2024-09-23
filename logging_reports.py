# SPDX-FileCopyrightText: 2024 Nico Hambauer
#
# SPDX-License-Identifier: MIT

import os
import re
from glob import glob

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score, mean_absolute_error, explained_variance_score, \
    max_error
from scipy.stats import rankdata


class Averager():
    def __init__(self, dir):
        super(Averager, self).__init__()
        self.dir = dir

    def avg_reg_report_mean_vals_with_std(self, model):

        csvs = sorted(glob(self.dir + f'{model}_Fold_*[0-9].csv'))
        print(self.dir)
        df = pd.concat([pd.read_csv(f, index_col=0) for f in csvs], ignore_index=False)

        df_std = df.std()
        df_mean = df.mean()
        df_std.to_csv(self.dir + f'{model}_Fold_std.csv', float_format='%.3f', index=True)
        df_mean.to_csv(self.dir + f'{model}_Fold_mean.csv', float_format='%.3f', index=True)

        out = df_mean.copy().astype(str)  # np.empty(df_mean.shape, dtype=object)
        for i in range(df_mean.shape[0]):
            out.iloc[i] = "{:.3f}".format(df_mean.values[i]) + u"\u00B1" + "{:.3f}".format(df_std.values[i])
        # df_out = pd.DataFrame(out)
        out.to_csv(self.dir + f'{model}_Fold_mean_std.csv', float_format='%.3f', index=True)

    def avg_reg_report_mean_vals(self, model):
        csvs = sorted(glob(self.dir + f'{model}_Fold_*[0-9].csv'))
        df = pd.concat([pd.read_csv(f, index_col=0) for f in csvs], ignore_index=False)
        df.to_csv(self.dir + f'{model}_Fold_concat.csv', index=True)
        df_mean = pd.DataFrame(df.mean())  # wrong: .groupby(level=0) since it is reg report
        df_mean.to_csv(self.dir + f'{model}_Fold_mean.csv', index=True)

    def avg_class_report_mean_vals_with_std(self, model):
        csvs = sorted(glob(self.dir + f'{model}_Fold_*[0-9].csv'))
        try:
            df = pd.concat([pd.read_csv(f, index_col=0) for f in csvs], ignore_index=False)
        except ValueError as e:
            print(e)
            print("Did you specify the correct models?")
            return

        df_std = df.groupby(level=0).std()
        df_mean = df.groupby(level=0).mean()
        df_std.to_csv(self.dir + f'{model}_Fold_std.csv', float_format='%.3f', index=True)
        df_mean.to_csv(self.dir + f'{model}_Fold_mean.csv', float_format='%.3f', index=True)

        out = df_mean.copy().astype(str)  # np.empty(df_mean.shape, dtype=object)
        for i in range(df_mean.shape[0]):
            for j in range(df_mean.shape[1]):
                out.iloc[i, j] = "{:.3f}".format(df_mean.values[i, j]) + u"\u00B1" + "{:.3f}".format(
                    df_std.values[i, j])
        df_out = pd.DataFrame(out)
        df_out.to_csv(self.dir + f'{model}_Fold_mean_std.csv', index=True)

    def avg_timings_mean_vals_with_std(self, model):

        csvs = sorted(glob(self.dir + f'Timing_{model}_Fold_*[0-9].csv'))
        # csvs = sorted(glob(self.current_dataset_model_dir + f'Timing_{model}_Fold_*.csv'))
        df_concat = pd.concat([pd.read_csv(f, index_col=0) for f in csvs], ignore_index=False)
        # df_concat.to_csv(self.dir + f'Timing_{model}_Fold_concat.csv', index=True)

        # df.timing = pd.to_datetime(df.timing).values.astype(np.int64)
        #
        # mean = df.groupby('model').mean()
        # may deviate at averaging due to overflows, but microseconds is reliable
        df_mean_timing = df_concat.groupby('model').mean()  # pd.to_datetime(.timing)
        df_std_timing = df_concat.groupby('model').std()
        # df_out = pd.DataFrame({'Average_Timing_Seconds': df_mean_timing}, index=[1])
        df_mean_timing.to_csv(self.dir + f'Timing_{model}_Fold_mean.csv', float_format='%.3f',
                              index=True)
        df_std_timing.to_csv(self.dir + f'Timing_{model}_Fold_std.csv', float_format='%.3f', index=True)

        df_mean_std_timing = pd.DataFrame(df_mean_timing.copy().astype(str))
        for i in range(df_mean_timing.shape[0]):
            for j in range(df_mean_timing.shape[1]):
                df_mean_std_timing.iloc[i, j] = "{:.3f}".format(
                    df_mean_timing.values[i, j]) + u"\u00B1" + "{:.3f}".format(
                    df_std_timing.values[i, j])

        df_mean_std_timing.to_csv(self.dir + f'Timing_{model}_Fold_mean_std.csv', float_format='%.3f', index=True)


class JournalLogger():
    """
    Used to log the results of the
    experiments in a journal.
    """

    def __init__(self):
        self.dir = None
        self.current_dataset_model_dir = None
        self.current_model_name = None

    def set_global_result_dir(self, dir='results/run_0'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.dir = dir

    def get_global_result_dir(self):
        self.assert_class_invariants()
        return self.dir

    def set_current_dataset_model_dir(self, dataset_name, model_name):
        """
        :param dataset_name: dataset_name
        :param model_name: model_name
        :return:
        """

        self.current_model_name = model_name

        self.current_dataset_model_dir = f'{self.dir}/{dataset_name}/{self.current_model_name}_Folds'

    def get_current_dataset_model_dir(self):
        """
        Returns Result path for Cross Validation
        """
        self.assert_class_invariants()

        return self.current_dataset_model_dir

    def log_classification_report(self, y_true, y_pred, k_fold=0, dataset=None):
        self.assert_class_invariants()
        labels = None
        target_names = None
        if dataset is not None:
            labels = dataset.labels
            target_names = dataset.target_names
        class_report = metrics.classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, labels=labels,
                                                     target_names=target_names)

        classification_report_df = pd.DataFrame(class_report).transpose()
        result_dir = self.get_current_dataset_model_dir()
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        classification_report_df.to_csv(f"{result_dir}/{self.current_model_name}_Fold_{k_fold + 1}.csv", index=True)

    def log_timing(self, timing, timing_mean_hpo, k_fold=0):
        self.assert_class_invariants()
        # For current dataset gam models save log loss as csv file
        timing_df = pd.DataFrame({'model': self.current_model_name, 'timing_best_model': timing,
                                  'mean_timing_all_hpo_configs': timing_mean_hpo}, index=[k_fold + 1])

        result_dir = self.get_current_dataset_model_dir()
        # and save
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        timing_df.to_csv(f'{result_dir}/Timing_{self.current_model_name}_Fold_{k_fold + 1}.csv', float_format='%.2f',
                         index=True)

    def log_regression_report(self, y_true, y_pred, k_fold=0):
        self.assert_class_invariants()
        rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
        mse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=True)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        # Also log r2 value
        r_2 = r2_score(y_true=y_true, y_pred=y_pred)
        # Also log explained variance score
        evs = explained_variance_score(y_true=y_true, y_pred=y_pred)
        # Also log the max error
        max_err = max_error(y_true=y_true, y_pred=y_pred)
        regression_report_df = pd.DataFrame(np.array([[rmse, mse, mae, evs, r_2, max_err]]),
                                            columns=['RMSE', 'MSE', 'MAE', 'EVS', 'R2', 'Max Error'])
        result_dir = self.get_current_dataset_model_dir()
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        regression_report_df.to_csv(f"{result_dir}/{self.current_model_name}_Fold_{k_fold + 1}.csv", index=True)

        result_dir = self.get_current_dataset_model_dir()
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        regression_report_df.to_csv(f"{result_dir}/{self.current_model_name}_Fold_{k_fold + 1}.csv", index=True)

    def log_roc_auc(self, y_true, y_pred_confidence, k_fold=0):
        """
        log the roc auc score right next to the classification report
        """
        self.assert_class_invariants()
        if type(y_pred_confidence) == np.array or type(y_pred_confidence) == np.ndarray:
            if y_pred_confidence.ndim > 1:
                if len(y_pred_confidence[0]) > 1:
                    y_pred_confidence = y_pred_confidence[:, 1]
                else:
                    y_pred_confidence = y_pred_confidence[:, 0]
        if type(y_pred_confidence) == list:
            if type(y_pred_confidence[0]) == list:
                if len(y_pred_confidence[0]) > 1:
                    y_pred_confidence = y_pred_confidence[:, 1]
        # print("shape pred after reshape", y_pred_confidence.ndim)

        roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred_confidence)

        result_dir = self.get_current_dataset_model_dir()
        class_report = pd.read_csv(f"{result_dir}/{self.current_model_name}_Fold_{k_fold + 1}.csv", index_col=0)
        class_report["ROC_AUC"] = None
        class_report.loc["accuracy", "ROC_AUC"] = roc_auc
        # report_df = pd.DataFrame(np.array([[roc_auc]]), columns=['ROC_AUC'])
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        class_report.to_csv(f"{result_dir}/{self.current_model_name}_Fold_{k_fold + 1}.csv", index=True)

    def assert_class_invariants(self):
        if self.dir is None:
            raise ValueError("Call set_gloabl_result_dir(dir) first to set the respective result logging dir")
        if self.current_dataset_model_dir is None:
            raise ValueError("Call set_datset_model_dir(dataset_name, model_name) first to set the respective result "
                             "logging dir for each dataset and model")
        if self.current_model_name is None:
            raise ValueError("Call set_datset_model_dir(dataset_name, model_name) first to set the respective "
                             "result logging dir for each dataset and model")


class TableForPaperAggregator():

    def __init__(self, dir='./results/run_0/', resultdir='./results/run_0/'):
        """

        :param dir: directory with slash at the end
        :param resultdir: resulting directory with slash at the end to put the table for paper
        """

        self.dir = dir
        self.resultdir = resultdir
        self.metric = None
        self.avg_metric = None

    def aggregate_timing_table_for_paper_csv(self, log_with_std=False, output_column_order=None, output_row_order=None):

        if not log_with_std:
            csvs = sorted(glob(self.dir + '*/*/' + f'Timing_*_Fold_mean.csv'))
        else:
            csvs = sorted(glob(self.dir + '*/*/' + f'Timing_*_Fold_mean_std.csv'))
        csvs = [c.replace('\\', '/') for c in csvs]

        dataset_names, model_names = self._get_dataset_model_reports_with_regex(csvs, timing=True)

        df_result_timing = pd.DataFrame()
        df_result_timing["dataset"] = list(dict.fromkeys(dataset_names))
        df_result_timing.set_index("dataset", inplace=True)

        for key in ['timing_best_model', 'mean_timing_all_hpo_configs']:
            for csv_report_path, dataset, model in zip(csvs, dataset_names, model_names):
                df_csv_timing = pd.read_csv(csv_report_path, index_col=0)
                df_result_timing.loc[dataset, model] = df_csv_timing.loc[df_csv_timing.index[0], key]

            if output_column_order:
                order = output_column_order
                df_result_timing = df_result_timing[order]

            if output_row_order:
                order = output_row_order
                # order the rows of the df according to the order of the list
                df_result_timing = df_result_timing.reindex(order)

            if log_with_std:
                df_result_timing.to_csv(self.resultdir + f'{key}_table_for_paper_mean_std.csv')
            else:
                df_result_timing.to_csv(self.resultdir + f'{key}_table_for_paper_mean.csv')
            print("Timing Table for Paper written to:", self.resultdir + f'{key}_table_for_paper.csv')

    def aggregate_table_for_paper(self, log_with_std=False, metric='f1-score', avg_metric='micro',
                                  output_column_order=None, output_row_order=None, table_name="table_for_paper"):
        """

        :param log_with_std:
        :param metric:  precision, recall, f1-score or ROC_AUC if classification and RMSE or MSE
        :param avg_metric: micro/accuracy, macro or weighted
        :param output_column_order:
        :param output_row_order:
        :param table_name:
        :return:
        """

        if metric == "ROC_AUC":
            avg_metric = "accuracy"  # ROC_AUC will be put next to the classification report in accuracy row

        self.metric = metric
        self.avg_metric = avg_metric


        if not log_with_std:
            csvs = sorted(glob(self.dir + '*/*/' + f'*_Fold_mean.csv'))
        else:
            csvs = sorted(glob(self.dir + '*/*/' + f'*_Fold_mean_std.csv'))
        csvs = [c.replace('\\', '/') for c in csvs]
        # filter out entries where Timing_ is in the path
        csvs = [c for c in csvs if 'Timing_' not in c]

        # print(csvs)
        dataset_names, model_names = self._get_dataset_model_reports_with_regex(csvs)
        # print(len(dataset_names), len(model_names))

        df_list = [pd.read_csv(f, index_col=0) for f in csvs]  # list of
        # dataset_names = d_n
        # print(df_list)
        df_result = pd.DataFrame()
        df_result["dataset"] = list(dict.fromkeys(dataset_names))
        df_result.set_index("dataset", inplace=True)
        print("Dataset:", dataset_names, "models:", model_names)
        for i, (dataset, model) in enumerate(zip(dataset_names, model_names)):
            metric_row = self._get_metric_row()
            if metric_row == 'accuracy' and 'accuracy' not in df_list[i].index:
                metric_row = 'micro'
                metric_column = self.metric
                df_result.loc[dataset, model] = df_list[i].loc[metric_row, metric_column]
            if self.metric in ['RMSE', 'MSE', 'MAE', 'EVS', 'R2', 'Max Error']:
                metric_row = self.metric
                # if dimension has 1 at the first place transpose the df_list[i]
                if df_list[i].shape[0] == 1:
                    df_list[i] = df_list[i].transpose()

                df_list[i].columns = df_list[i].columns.astype(str)

                df_result.loc[dataset, model] = df_list[i].loc[metric_row, "0"]
            elif self.metric in ['precision', 'recall', 'f1-score', 'ROC_AUC']:
                metric_column = self.metric
                df_result.loc[dataset, model] = df_list[i].loc[metric_row, metric_column]

        print("Table for Paper written to:", self.resultdir + f'/{table_name}.csv')
        if output_column_order:
            order = output_column_order
            df_result = df_result[order]

        if output_row_order:
            order = output_row_order
            # order the rows of the df according to the order of the list
            df_result = df_result.reindex(order)

        if not log_with_std:
            if self.metric in ['precision', 'recall', 'f1-score', 'ROC_AUC']:
                values = -df_result.values
            else:
                values = df_result.values
            ranks = pd.DataFrame(rankdata(values, axis=1), index=df_result.index, columns=df_result.columns)
            ranks.to_csv(self.resultdir + f'/{table_name}_ranks.csv')


        df_result.to_csv(self.resultdir + f'/{table_name}.csv')

    def _get_dataset_model_reports_with_regex(self, csvs, timing=False):
        csvs = [c.replace('\\', '/') for c in csvs]
        model_names = []
        for csv in csvs:
            for match in re.finditer(r'[A-Z_a-z]+_Fold_mean', csv):
                m = match.group()[:-10]
                if timing and 'Timing' in m:
                    m = m.replace('Timing_', '')
                    model_names.append(m)  # append only timing csvs
                elif 'Timing' not in m:
                    model_names.append(m)  # append only non timing csvs
        dataset_names = []
        for csv in csvs:
            for match in re.finditer(rf'{self.dir}[a-z]+/', csv):
                d = match.group()[len(self.dir):-1]
                dataset_names.append(d)

        return dataset_names, model_names

    def _get_metric_row(self):
        row = None
        if self.avg_metric == 'micro' or self.avg_metric == 'accuracy':
            row = 'accuracy'
        elif self.avg_metric == 'macro':
            row = 'macro avg'
        elif self.avg_metric == 'weighted':
            row = 'weighted avg'
        return row


if __name__ == '__main__':

    task = 'regression'  # 'classification'

    dir = f'./results/pre-study/mgcv-ts-20/{task}/'

    avg = Averager(dir)
    if task == 'classification':
        traditional_models_to_run = [
            # 'LR',
            # 'DT',
            # 'RF',
            # 'XGB',
            # 'MLP'
            # "TABNET",
            # "CATBOOST",
        ]
    else:
        traditional_models_to_run = [
            # 'ELASTICNET',
            # 'DT',
            # 'RF',
            # 'XGB',
            # 'MLP',
            # "TABNET",
            # "CATBOOST",
        ]

    gam_models_to_run = [
        # 'PYGAM',
        # 'EBM',
        # 'NAM',
        # 'GAMINET',
        # 'EXNN',
        # 'IGANN'
        "MGCV_SPLINE"
    ]

    all_models = traditional_models_to_run + gam_models_to_run

    classification_datasets = [
        'water',
        'stroke',
        'weather',
        'adult',
        'telco',
        'college',
        'fico',
        'bank',
        'airline',
        'compas'
    ]

    regression_datasets = [
        'car',
        'crab',
        'diamond',
        'medical',
        'productivity',
        'student',
        'wine',
        'crimes',
        'bike',
        'housing'
    ]

    if task == 'classification':

        for dataset in classification_datasets:
            for model in all_models:
                avg.dir = f'{dir}{dataset}/{model}_Folds/'
                avg.avg_class_report_mean_vals_with_std(model)
                avg.avg_timings_mean_vals_with_std(model)

        agg = TableForPaperAggregator(dir=dir, resultdir=dir)

        agg.aggregate_timing_table_for_paper_csv(log_with_std=True,
                                                 # output_column_order=['PYGAM', 'EBM', 'NAM', 'GAMINET',
                                                 #                      'EXNN', 'IGANN', 'LR', 'DT', 'RF', 'XGB', 'MLP'],
                                                 output_row_order=['college', 'water', 'stroke', 'telco',
                                                                   'compas', 'fico',
                                                                   'adult', 'bank', 'airline', 'weather'],
                                                 )
        agg.aggregate_table_for_paper(log_with_std=True,
                                      # output_column_order=['PYGAM', 'EBM', 'NAM', 'GAMINET',
                                      #                      'EXNN', 'IGANN', 'LR', 'DT', 'RF', 'XGB', 'MLP'],
                                      output_row_order=['college', 'water', 'stroke', 'telco', 'compas', 'fico', 'adult', 'bank',
                                                        'airline', 'weather'],
                                      table_name=f'{task}_table', # 'argsort_classification_table',
                                      metric = 'ROC_AUC',
                                      )
    elif task == 'regression':

        for dataset in regression_datasets:
            for model in all_models:
                avg.dir = f'{dir}{dataset}/{model}_Folds/'
                avg.avg_reg_report_mean_vals_with_std(model)
                avg.avg_timings_mean_vals_with_std(model)

        metric = 'RMSE'
        #
        agg = TableForPaperAggregator(dir=dir, resultdir=dir)
        #
        agg.aggregate_table_for_paper(log_with_std=True,
                                      # output_column_order=['PYGAM', 'EBM', 'NAM', 'GAMINET',
                                      #                      'EXNN', 'IGANN',
                                      # 'ELASTICNET', 'DT', 'RF', 'XGB', 'MLP'],
                                      # output_column_order=['ELASTICNET', 'DT', 'RF', 'XGB', 'MLP'],

                                      output_row_order=['car', 'student', 'productivity', 'medical', 'crimes', 'crab', 'wine', 'bike', 'housing', 'diamond'],
                                      table_name=f'{task}_table_{metric.lower()}', # rank_...
                                      metric=metric,
                                      avg_metric=None #  None for regression
                                      )

        agg.aggregate_timing_table_for_paper_csv(log_with_std=True,
                                                 # output_column_order=['PYGAM', 'EBM', 'NAM', 'GAMINET', 'EXNN', 'IGANN',
                                                 #                      'ELASTICNET', 'DT', 'RF', 'XGB', 'MLP'],
                                                 output_row_order=['car', 'student', 'productivity', 'medical',
                                                                   'crimes', 'crab', 'wine', 'bike', 'housing',
                                                                   'diamond'],
                                                 )
