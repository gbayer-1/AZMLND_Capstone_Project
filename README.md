*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Machine Learning Engineer with Microsoft Azure: Capstone Project: Predicting Heart Failure

*TODO:* Write a short introduction to your project.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.
The dataset I'm using for this project is the Heart Failure Prediction Dataset from kaggle.

fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved [2021-10-18] from https://www.kaggle.com/fedesoriano/heart-failure-prediction.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

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
  - resting blood pressure [mm Hg]
  - resting ECG results in three categories (Normal,  ST: ST-T wave abnormality, LVH : left ventricular hypertrophy)
  - oldpeak: the depression between S and T peak
  - maximum heartrate under cardiac stress
  - exercise-induced angina
  - ST-slope of te peak exercise in three categories (Up, flat and down)

### Access
*TODO*: Explain how you are accessing the data in your workspace.
Since I'm using the same dataset for an AutoML run and a HyperDrive Experiment, I defined the access to the data in the train.py script, so it can be used in both notebooks.

### PreProcessing
The preprocessing of the data is defined in the prepare_data function in the train.py script.
Some categorical features have to be preprocessed before they can be handled by the machine learning models.
- The columns Sex and ExerciseAngina are binary coded, with 1 for "male"/"yes" and 0 for "female"/"no" respectively.
- The column ST_Slope is numerically encoded with 1 for "Up", 0 for "Flat" and -1 for "Down".
- The columns for the ChestPainType and RestingECG features are one_hot encoded for the model.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
