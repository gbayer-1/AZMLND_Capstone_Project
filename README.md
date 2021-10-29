# Machine Learning Engineer with Microsoft Azure Capstone Project: <br> :heart: Predicting Heart Failure :broken_heart:

> Cardiovascular diseases (CVDs) are the leading cause of death globally.
> An estimated 17.9 million people died from CVDs in 2019, representing 32% of all global deaths. Of these deaths, 85% were due to heart attack and stroke.
> It is important to detect cardiovascular disease as early as possible so that management with counselling and medicines can begin.
> <br> [WHO fact sheet on cardiovascular diseases](https://www.who.int/en/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds) Retrieved: 2021-10-28)

Due to the large impact of CVD I am using this capstone project to train a machine learning model to help doctors diagnose patients with CVD. This should be achieved, while using only a small set of diagnostic data, which can be easily obtained by a medical professional. Early detection and treatment of a CVD is highly favorable for the survival of the patients.

To tackle this task, I'm using the free heart-failure-prediction dataset from [kaggle.com](http://www.kaggle.com/) (see section [Dataset](#dataset)). I'm using two approaches to train a model: On the one hand I use the automated machine learning feature from Azure called AutoML (see section [Automated ML](#automated-ml)), which trains many different model types on the dataset. On the other hand I use the AzureML HyperDrive package to automatically tune hyperparameters of a RandomForestClassifier model (see section [Hyperparameter Tuning](#hyperparameter-tuning)). I then deploy the best model to a WebService and interact with it (see section [Model Deployment](#model-deployment)). Additionally I'm giving a short excursion on how to convert the model into the ONNX-framework and on how to monitor the deployed endpoint using logging (see section [Standout Suggestions](#standout-suggestions)).

## Project Set Up and Installation
<img src="./screenshots/screenshots_firsttry/Screenshot 2021-10-25 130458_upload_folder.png" width=300 align="left"/>
<img src="./screenshots/notebook_section.png" width=200 align="right" />

This project consists of two jupyter notebooks [automl.ipynb](automl.ipynb) and [hyperparameter_training.ipynb](hyperparameter_training.ipynb), one python file [train.py](train.py) containing functions to clean and read in the data, as well as the training algorithm for the model to be used in the HyperDrive experiment and the folder [data/](data/) containing the dataset in .csv format.
In order to run both notebooks, the whole project folder, including the subfolder data has to be uploaded into the Azure Machine Learning Studio Notebook section.

The notebook section should look like this ➡️

Please make sure to adjust the `subscription_id`,`ressource_group` and `workspace_name` in the [config.json](./config.json) according to your subscription

## Dataset

### Overview
The dataset I'm using for this project is the Heart Failure Prediction Dataset from kaggle.

fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved [2021-10-18] from https://www.kaggle.com/fedesoriano/heart-failure-prediction.

I chose this dataset, since it is a quite comprehensive and balanced dataset, with features that are mostly comprehensible for non-professionals too.

### Task

The task with this dataset is to predict whether a person will develop a heart disease with a set of 11 diagnostic features.
This dataset is a combination of five independent heart disease datasets containing 918 observations of patients. The target column "Heart Disease" is nearly balanced in this dataset with 510 patients with and 408 patients without cardiovascular diseases.

The features in this dataset are:
- General information
  - Age of the patient in years; with the youngest patient being 28 and the oldest 77 years old
  - Sex of the patient; the majority of the patients (79%) being male
- blood tests
  - Cholesterol in serum [mm/dl]; an indicator for ateriosclerosis
  - fasting blood sugar level; a boolean value if the blood sugar is elevated (>120mg/dl) or not, which is an indicator for diabetes
- medical history
  - type of chest pain the patient is experiencing in four categories (TA: typical angina, ATA: atypical angina, NAP: non-anginal pain, ASY: asymptomatic)
- ECG and cardiac stress testing
  [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/SinusRhythmLabels.svg/608px-SinusRhythmLabels.svg.png" width=200 align="right">](https://en.wikipedia.org/wiki/File:SinusRhythmLabels.svg#filelinks)

  - resting blood pressure [mm Hg]
  - resting ECG results in three categories (Normal,  ST: ST-T wave abnormality, LVH : left ventricular hypertrophy)
  - oldpeak: the depression between S and T peak in the ECG
  - maximum heartrate under cardiac stress
  - exercise-induced angina
  - ST-slope in the ECG of the peak exercise in three categories (up, flat and down) 

### Access
Since I'm using the same dataset for an AutoML run and a HyperDrive Experiment, I defined the access to the data in the function `read_data()` in the [train.py](train.py) script, so it can be used in both notebooks.

To be used in the Azure Machine Learning Studio the data has to be uploaded and registered in the Workspace. The dataset will be available by the name "heart_disease_data". First the function `read_data()` checks, whether a dataset of this name is already registered in the workspace `ws` with `ws.datasets["heart_diseases_data"]`. If it is found, then the function will reuse this dataset.

If the dataset is not found, the function first accesses the default datastore of the workspace, and uploads the contents of the data folder into a folder named "data" in the datastore
```
datastore = ws.get_default_datastore()
datastore.upload('./data', overwrite=True, target_path='data')
```
Then a Tabular Dataset is created using `from_delimited_files()`. I could already register this dataset, but I want to use my own preprocessing algorithm to clean the data, therefore I call the function `prepare_data()` (see [section PreProcessing](#preprocessing)).
The output of this function is a cleaned and ready to use pandas dataframe. I register this dataframe as `TabularData` in my workspace with the name "heart_disease_data".
```
dataset = Dataset.Tabular.register_pandas_dataframe(df, datastore, show_progress=True,
                             name="heart_disease_data", description=description_text)
```

### PreProcessing
The preprocessing of the data is defined in the `prepare_data` function in the [train.py](train.py) script.
Some categorical features have to be preprocessed before they can be handled by the machine learning models.
- The columns Sex and ExerciseAngina are binary coded, with 1 for "male"/"yes" and 0 for "female"/"no" respectively.
- The column ST_Slope is numerically encoded with 1 for "Up", 0 for "Flat" and -1 for "Down".
- The columns for the ChestPainType and RestingECG features are one_hot encoded for the model.

## Automated ML
For a detailed description of the AutoML configuration please refer to the incode documentation in [automl.ipynb](automl.ipynb).<br>
I'm running a classification task on the dataset without featurization using a 5-fold crossvalidation and the primary metric accuracy.

### Results
You can find an overview and discussion about the models trained by the AutoML run and a detailed description of the best model of this run in [automl.ipynb](automl.ipynb).<br>
The best model of the AutoML run is a VotingEnsemble containing the tree-based-models XGBoostClassifier and LightGBM. It has an accuracy of `0.889` and a precision of `0.892`.

<img src="./screenshots/AutoML_runwidget.png" />

A description on further improvements can also be found in the [notebook](automl.ipynb).

## Hyperparameter Tuning
For a detailed description of the hyperdrive configuration please refer to the incode documentation in [hyperparameter_tuning.ipynb](hyperparameter_tuning.ipynb). <br>
I chose a RandomForestClassifier model for this problem, since the input data is a mix of categorical und numerical data on which tree-based models tend to perform well. The HyperDrive Run tunes three parameters of this model on the primary metric accuracy: the number of trees in the forest, the maximum depth of a tree and the minimum number of samples required to split a node in the tree. The model training is defined in [train.py](train.py). Please note, that I log more than the primary metric of this model for each run. I convert the trained models into ONNX-Framework as well (see section [Standout Suggestions](#standout-suggestions)). 

### Results
You can find an overview and discussion about the models trained by the Hyperparameter tuning and a detailed description of the best model of this run in [hyperparameter_tuning.ipynb](hyperparameter_tuning.ipynb). <br>
The best RandomForestClassifier model of this run consists of 1416 trees with a maximum depth of 100 and at least 2 samples inside one leaf. It has an accuracy of $0.922$ and a precision of $0.896$.

<img src="./screenshots/hyperdrive_rundetails_1.png" />

A description on further improvements can also be found in the [notebook](hyperparameter_tuning.ipynb).

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
