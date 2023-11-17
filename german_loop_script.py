import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
import os
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from autoimpute.imputations import SingleImputer
from autoimpute.imputations import MultipleImputer
from numpy import nan
from numpy import isnan
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset
from aif360.explainers import MetricTextExplainer
from aif360.metrics import Metric, DatasetMetric, utils
from sklearn import preprocessing
from typing import List, Union, Dict
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def fair_metrics(dataset, pred, pred_is_dataset=False):
    if pred_is_dataset:
        dataset_pred = pred
    else:
        dataset_pred = dataset.copy()
        dataset_pred.labels = pred
    # Checking if there exists a dataset with only predictions in the previous condition
    cols = ['statistical_parity_difference', 'equal_opportunity_difference', 'average_abs_odds_difference',  'disparate_impact', 'theil_index']
    obj_fairness = [[0 ,0 ,0 ,1 ,0]]

    fair_metrics = pd.DataFrame(data=obj_fairness, index=['objective'], columns=cols)

    for attr in dataset_pred.protected_attribute_names:
        idx = dataset_pred.protected_attribute_names.index(attr)
        privileged_groups =  [{attr :dataset_pred.privileged_protected_attributes[idx][0]}]
        unprivileged_groups = [{attr :dataset_pred.unprivileged_protected_attributes[idx][0]}]

        # We need to use Classification Metric for calculating 3 Metrics (Equal Oppr, Theil Index & Avg Odds Diff) & BinaryLabel Dataset Metric for the rest
        classified_metric = ClassificationMetric(dataset,
                                                 dataset_pred,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

        metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)

        acc = classified_metric.accuracy()

        row = pd.DataFrame([[metric_pred.mean_difference(),
                             classified_metric.equal_opportunity_difference(),
                             classified_metric.average_abs_odds_difference(),
                             metric_pred.disparate_impact(),
                             classified_metric.theil_index()]],
                           columns  = cols,
                           index = [attr]
                           )
        fair_metrics = fair_metrics.append(row)

    fair_metrics = fair_metrics.replace([-np.inf, np.inf], 2)

    return fair_metrics



def get_fair_metrics(data, model, plot=True, model_aif=False):
    pred = model.predict(data).labels if model_aif else model.predict(data.features)
    fair = fair_metrics(data, pred)
    return fair


def nested_loop(iterable, cols1, col2, local_data_train, data_orig_test, percentage_list, imputation_types, imputation_type2, Classification_Models, results_df, privileged_sex, privileged_age):
    results_df_names = ['dataset_name', 'num_columns_imputed', 'percentage_deleted', 'imputation_strategy',
                        'repetition',
                        'classification_algorithm', 'accuracy', 'auc', 'F1_score', 'Sensitivity', 'Specificity',
                        'Statistical_Parity_Race', 'Statistical_Parity_Sex', 'Equal_Oppr_diff_Race',
                        'Equal_Oppr_diff_Sex',
                        'average_abs_odds_diff_Race', 'average_abs_odds_diff_Sex', 'Disparate_Impact_Race',
                        'Disparate_Impact_Sex', 'Theil_Index_Race', 'Theil_Index_Sex']
    results_df = pd.DataFrame(columns=results_df_names)

    for i in range(1, len(cols1) + 1):
        random_columns = np.random.choice(cols1, i,
                                          replace=False)  # Selecting columns at random, excluding Sensitive attributes
        local_data = local_data_train.copy(deep=True)

        for percentage in percentage_list:
            remove_n = int(round(local_data.shape[0] * percentage, 0))

            for column_name in random_columns:
                drop_indices = np.random.choice(local_data.index, remove_n, replace=False)
                local_data.loc[
                    drop_indices, column_name] = np.nan  # Removing a selected percent of data from each of the selected columns

            for imputer in imputation_types:
                if imputer in imputation_type2:
                    local_data_final = imputer.fit_transform(
                        local_data)  # Imputing the missing values using a dataframe
                else:
                    values = local_data.values
                    imputed_values = imputer.fit_transform(values)
                    local_data_final = pd.DataFrame(data=imputed_values, columns=col2)
                # Imputing the missing values after transforming the dataframe to a Numpy Array and then converting it back to a dataframe

                # Converting the imputed dataframe back to a Standard Dataset using AIf 360
                local_data_final2 = StandardDataset(local_data_final,
                                                    label_name='credit',
                                                    favorable_classes=[1],
                                                    protected_attribute_names=['sex', 'Age_Metric'],
                                                    privileged_classes=[privileged_sex, privileged_age])

                for classifier in Classification_Models:
                    log_reg_fit = classifier.fit(local_data_final2.features, local_data_final2.labels.ravel(),
                                                 sample_weight=local_data_final2.instance_weights)
                    X_test_credit = data_orig_test.features
                    y_test_credit = data_orig_test.labels.ravel()
                    Germancredit_pred_final = log_reg_fit.predict(X_test_credit)
                    Germancredit_pred_proba_final = log_reg_fit.predict_proba(X_test_credit)[:, 1]
                    Accuracy = metrics.accuracy_score(y_test_credit, Germancredit_pred_final)
                    AUC = roc_auc_score(y_test_credit, Germancredit_pred_proba_final)
                    F1_score = metrics.f1_score(y_test_credit, Germancredit_pred_final)
                    confusion_matrix_pre = metrics.confusion_matrix(y_test_credit, Germancredit_pred_final)
                    TP = confusion_matrix_pre[1, 1]
                    TN = confusion_matrix_pre[0, 0]
                    FP = confusion_matrix_pre[0, 1]
                    FN = confusion_matrix_pre[1, 0]
                    Sensitivity = TP / (TP + FN)
                    Specificity = TN / (TN + FP)
                    fair_final = get_fair_metrics(data_orig_test, log_reg_fit)
                    Statistical_Parity_Sex = fair_final.iloc[1, 0]
                    Statistical_Parity_Age = fair_final.iloc[2, 0]
                    Equal_Oppr_diff_Sex = fair_final.iloc[1, 1]
                    Equal_Oppr_diff_Age = fair_final.iloc[2, 1]
                    average_abs_odds_diff_Sex = fair_final.iloc[1, 2]
                    average_abs_odds_diff_Age = fair_final.iloc[2, 2]
                    Disparate_Impact_Sex = fair_final.iloc[1, 3]
                    Disparate_Impact_Age = fair_final.iloc[2, 3]
                    Theil_Index_Sex = fair_final.iloc[1, 4]
                    Theil_Index_Age = fair_final.iloc[2, 4]
                    if imputer.__class__.__name__ == 'KNNImputer':
                        imputation_strategy = imputer.__class__.__name__
                    elif imputer.__class__.__name__ == 'IterativeImputer':
                        imputation_strategy = imputer.__class__.__name__
                    else:
                        imputation_strategy = imputer.strategy

                    # Storing the fairness and classification metrics in a dataframe
                    new_row = ['German', i, percentage, imputation_strategy, 1, classifier.__class__.__name__, Accuracy,
                               AUC, F1_score, Sensitivity, Specificity,
                               Statistical_Parity_Sex, Statistical_Parity_Age, Equal_Oppr_diff_Sex, Equal_Oppr_diff_Age,
                               average_abs_odds_diff_Sex, average_abs_odds_diff_Age, Disparate_Impact_Sex,
                               Disparate_Impact_Age,
                               Theil_Index_Sex, Theil_Index_Age]
                    results_df.loc[len(results_df)] = new_row
    return results_df


def nested_loop2(iterable, cols1, col2, local_data_train, data_orig_test, percentage_list, imputation_types, imputation_type2, Classification_Models, results_df5, privileged_sex, privileged_age):

    results_df_names = ['dataset_name', 'num_columns_imputed', 'percentage_deleted', 'imputation_strategy',
                        'repetition',
                        'classification_algorithm', 'accuracy', 'auc', 'F1_score', 'Sensitivity', 'Specificity',
                        'Statistical_Parity_Race', 'Statistical_Parity_Sex', 'Equal_Oppr_diff_Race',
                        'Equal_Oppr_diff_Sex',
                        'average_abs_odds_diff_Race', 'average_abs_odds_diff_Sex', 'Disparate_Impact_Race',
                        'Disparate_Impact_Sex', 'Theil_Index_Race', 'Theil_Index_Sex']
    results_df5 = pd.DataFrame(columns=results_df_names)

    for i in range(1, len(cols1) + 1):
        random_columns = np.random.choice(cols1, i,
                                          replace=False)  # Selecting columns at random, excluding Sensitive attributes
        local_data = local_data_train.copy(deep=True)

        for percentage in percentage_list:
            remove_n = int(round(local_data.shape[0] * percentage, 0))

            for column_name in random_columns:
                drop_indices = np.random.choice(local_data.index, remove_n, replace=False)
                local_data.loc[
                    drop_indices, column_name] = np.nan  # Removing a selected percent of data from each of the selected columns

            for imputer in imputation_types:
                if imputer in imputation_type2:
                    local_data_final = imputer.fit_transform(
                        local_data)  # Imputing the missing values using a dataframe
                else:
                    values = local_data.values
                    imputed_values = imputer.fit_transform(values)
                    local_data_final = pd.DataFrame(data=imputed_values, columns=col2)
                # Imputing the missing values after transforming the dataframe to a Numpy Array and then converting it back to a dataframe

                # Converting the imputed dataframe back to a Standard Dataset using AIf 360
                local_data_final2 = StandardDataset(local_data_final,
                                                    label_name='credit',
                                                    favorable_classes=[1],
                                                    protected_attribute_names=['sex', 'Age_Metric'],
                                                    privileged_classes=[privileged_sex, privileged_age])

                for classifier in Classification_Models:
                    log_reg_fit = classifier.fit(local_data_final2.features, local_data_final2.labels.ravel(),
                                                 sample_weight=local_data_final2.instance_weights)
                    X_test_credit = data_orig_test.features
                    y_test_credit = data_orig_test.labels.ravel()
                    Germancredit_pred_final = log_reg_fit.predict(X_test_credit)
                    Germancredit_pred_proba_final = log_reg_fit.predict_proba(X_test_credit)[:, 1]
                    Accuracy = metrics.accuracy_score(y_test_credit, Germancredit_pred_final)
                    AUC = roc_auc_score(y_test_credit, Germancredit_pred_proba_final)
                    F1_score = metrics.f1_score(y_test_credit, Germancredit_pred_final)
                    confusion_matrix_pre = metrics.confusion_matrix(y_test_credit, Germancredit_pred_final)
                    TP = confusion_matrix_pre[1, 1]
                    TN = confusion_matrix_pre[0, 0]
                    FP = confusion_matrix_pre[0, 1]
                    FN = confusion_matrix_pre[1, 0]
                    Sensitivity = TP / (TP + FN)
                    Specificity = TN / (TN + FP)
                    fair_final = get_fair_metrics(data_orig_test, log_reg_fit)
                    Statistical_Parity_Sex = fair_final.iloc[1, 0]
                    Statistical_Parity_Age = fair_final.iloc[2, 0]
                    Equal_Oppr_diff_Sex = fair_final.iloc[1, 1]
                    Equal_Oppr_diff_Age = fair_final.iloc[2, 1]
                    average_abs_odds_diff_Sex = fair_final.iloc[1, 2]
                    average_abs_odds_diff_Age = fair_final.iloc[2, 2]
                    Disparate_Impact_Sex = fair_final.iloc[1, 3]
                    Disparate_Impact_Age = fair_final.iloc[2, 3]
                    Theil_Index_Sex = fair_final.iloc[1, 4]
                    Theil_Index_Age = fair_final.iloc[2, 4]
                    if imputer.__class__.__name__ == 'KNNImputer':
                        imputation_strategy = imputer.__class__.__name__
                    elif imputer.__class__.__name__ == 'IterativeImputer':
                        imputation_strategy = imputer.__class__.__name__
                    else:
                        imputation_strategy = imputer.strategy

                    # Storing the fairness and classification metrics in a dataframe
                    new_row = ['German_Cat', i, percentage, imputation_strategy, iterable, classifier.__class__.__name__, Accuracy,
                               AUC, F1_score, Sensitivity, Specificity,
                               Statistical_Parity_Sex, Statistical_Parity_Age, Equal_Oppr_diff_Sex, Equal_Oppr_diff_Age,
                               average_abs_odds_diff_Sex, average_abs_odds_diff_Age, Disparate_Impact_Sex,
                               Disparate_Impact_Age,
                               Theil_Index_Sex, Theil_Index_Age]
                    results_df5.loc[len(results_df5)] = new_row
    return results_df5
