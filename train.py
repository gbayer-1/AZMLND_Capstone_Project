# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 10:35:42 2021

@author: GABRIFR
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from azureml.core.run import Run
from azureml.core import Dataset
import argparse
import numpy as np
from azureml.core import Workspace
#import os

def prepare_data(df):
    '''
    Some categorical features have to be preprocessed before they can be handled by the machine learning models.

    The columns Sex and ExerciseAngina are binary coded, with 1 for "male"/"yes" and 0 for "female"/"no" respectively.
    The column ST_Slope is numerically encoded with 1 for "Up", 0 for "Flat" and -1 for "Down".
    The columns for the ChestPainType and RestingECG features are one_hot encoded for the model.

    '''
    df['Sex'] = df.Sex.apply(lambda s: 1 if s=='M' else 0)
    df['ExerciseAngina'] = df.ExerciseAngina.apply(lambda s: 1 if s is True else 0)
    
    one_hot_cols = ["ChestPainType", "RestingECG"]
    for col in one_hot_cols:
        tmp_col = pd.get_dummies(df[col], prefix=col)
        df.drop(col, inplace=True, axis=1)
        df = df.join(tmp_col)
    
    ST_Slope = {'Up': 1, 'Flat': 0, 'Down': -1}
    df["ST_Slope"] = df.ST_Slope.map(ST_Slope)
    return df

def test_func():
    ws = Workspace.from_config()
    default_datastore = ws.get_default_datastore()
    default_datastore.upload('./data')

# TODO change data read in
def read_data():
    '''
    This function uploads the local file heart.csv into the default datastore
    of the Azure Workspace and then creates and registers a TabularDataset 
    from it.

    Returns
    -------
    dataset : azureml.data.dataset_factory.TabularDatasetFactory

    '''
    
    ws = Workspace.from_config()
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
   
        dataset = Dataset.Tabular.from_delimited_files(path = [(datastore, 'data/heart.csv')],
                                                        validate=True,
                                                            infer_column_types=True,
                                                            separator=',',
                                                            header=True,
                                                            support_multi_line=False,
                                                            empty_as_string=False,
                                                            encoding="utf8")
        df = prepare_data(dataset.to_pandas_dataframe())
        dataset = Dataset.Tabular.register_pandas_dataframe(df,datastore, show_progress=True,
                             name=data_name, description=description_text)
    return dataset

##ds = read_data()
#df = ds.to_pandas_dataframe()
#X = prepare_data(df)
#y = X.pop("HeartDisease")

#x_train, x_test, y_train, y_test = train_test_split(X, y)

#run = Run.get_context()

def main():
    # add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
