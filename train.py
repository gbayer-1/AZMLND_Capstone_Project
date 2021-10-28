# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 10:35:42 2021

@author: GABRIFR
"""

from azureml.core import Workspace, Dataset
from azureml.core.run import Run
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_score, roc_auc_score
import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


run = Run.get_context()
if hasattr(run, 'experiment'):
    # get the workspace from the Run object, if this script is called during the hyperdrive run
    ws = run.experiment.workspace
else:
    # get the workspace from the config file, if this script is called from the notebook
    ws = Workspace.from_config()


def prepare_data(df):
    '''
    Some categorical features have to be preprocessed before they can be handled by the machine learning models.

    The columns Sex and ExerciseAngina are binary coded, with 1 for "male"/"yes" and 0 for "female"/"no" respectively.
    The column ST_Slope is numerically encoded with 1 for "Up", 0 for "Flat" and -1 for "Down".
    The columns for the ChestPainType and RestingECG features are one_hot encoded for the model.

    '''
    df['Sex'] = df.Sex.apply(lambda s: 1 if s == 'M' else 0)
    df['ExerciseAngina'] = df.ExerciseAngina.apply(lambda s: 1 if s is True else 0)

    one_hot_cols = ["ChestPainType", "RestingECG"]
    for col in one_hot_cols:
        tmp_col = pd.get_dummies(df[col], prefix=col)
        df.drop(col, inplace=True, axis=1)
        df = df.join(tmp_col)

    ST_Slope = {'Up': 1, 'Flat': 0, 'Down': -1}
    df["ST_Slope"] = df.ST_Slope.map(ST_Slope)
    return df


def read_data():
    '''
    This function uploads the local file heart.csv into the default datastore
    of the Azure Workspace and then creates and registers a TabularDataset
    from it.

    '''
    data_name = "heart_disease_data"
    description_text = "heart_failure_prediction_dataset from kaggle"

    # Test, whether the dataset is already in the workspace
    if data_name in ws.datasets.keys():
        print('found existing dataset. use it')
        dataset = ws.datasets[data_name]
    else:
        # Create the dataset
        datastore = ws.get_default_datastore()
        datastore.upload('./data', overwrite=True, target_path='data')

        dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, 'data/heart.csv')],
                                                       validate=True,
                                                       infer_column_types=True,
                                                       separator=',',
                                                       header=True,
                                                       support_multi_line=False,
                                                       empty_as_string=False,
                                                       encoding="utf8")
        # get pandas dataframe from the Tabular Dataset as input into prepare_data
        df = prepare_data(dataset.to_pandas_dataframe())
        # finally register the Tabular Dataset from the cleaned pandas dataframe
        dataset = Dataset.Tabular.register_pandas_dataframe(df, datastore, show_progress=True,
                                                            name=data_name, description=description_text)
    return dataset


def main():
    '''
    Trains a RandomForestClassifier model, logs metrics of the trained model and saves the model in pkl and onnx format.
    '''
    # add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="The number of trees in the forest.")
    parser.add_argument('--max_depth', type=int, default=None, help="The maximum depth of the tree")
    parser.add_argument('--min_samples_split', type=int, default=2,
                        help="The minimum number of samples required to split an internal node")
    args = parser.parse_args()

    # get the data, and use train_test_split to get training and test input for the model
    X = read_data().to_pandas_dataframe()
    y = X.pop("HeartDisease")

    x_train, x_test, y_train, y_test = train_test_split(X, y)

    # train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=args.n_estimators,
                                   max_depth=args.max_depth,
                                   min_samples_split=args.min_samples_split).fit(x_train, y_train)

    # log parameters of the trained model
    run.log("No. of trees:", np.float(args.n_estimators))
    run.log("Max depth:", np.int(args.max_depth))
    run.log("Min samples split:", np.int(args.min_samples_split))

    # log the feature importance for this model
    feature_importance = model.feature_importances_
    feature_list = model.feature_names_in_
    for index in range(len(feature_importance)):
        run.log_row("feature importance", name=feature_list[index], importance=feature_importance[index])

    # log model metrics
    accuracy = model.score(x_test, y_test)
    y_pred = model.predict(x_test)

    run.log("Accuracy", np.float(accuracy))
    run.log("F1 Score", np.float(f1_score(y_test, y_pred)))
    run.log("precision", np.float(precision_score(y_test, y_pred)))
    run.log("AUC", np.float(roc_auc_score(y_test, y_test)))

    conf_matrix = confusion_matrix(y_test, y_pred)
    cmtx = {
        "schema_type": "confusion_matrix",
        "schema_version": "1.0.0",
        "data": {"class_labels": ["0", "1"],
                 "matrix": [[int(y) for y in x] for x in conf_matrix]}}
    run.log_confusion_matrix("confusion matrix", cmtx)

    # save model als pickle
    os.makedirs('outputs', exist_ok=True)
    with open("outputs/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # save model as onnx
    initial_type = [('X', FloatTensorType([None, x_train.shape[1]]))]
    onnx = convert_sklearn(model, initial_types=initial_type)
    with open('outputs/hyperdrive_model.onnx', "wb") as f:
        f.write(onnx.SerializeToString())


if __name__ == '__main__':
    main()
