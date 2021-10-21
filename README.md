*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Machine Learning Engineer with Microsoft Azure: Capstone Project: Predicting Heart Failure

*TODO:* Write a short introduction to your project.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

This project consists of two jupyter notebooks `automl.ipynb` and `hyperprameter_training.ipynb`, one python file `train.py` containing function to clean and read in the data, as well as the training algorithm for the model to be used in the HyperDrive experiment and the folder `data\` containing the dataset in .csv format.
In order to run both notebooks, the whole project folder, including the subfolder data has to be uploaded into the Azure Machine Learning Studio Notebook section.

*TODO*: Make screenshot of the upload

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
Since I'm using the same dataset for an AutoML run and a HyperDrive Experiment, I defined the access to the data in the function `read_data` in the train.py script, so it can be used in both notebooks.

To be used in the Azure Machine Learning Studio the data has to be uploaded and registered in the Workspace. The dataset will be available by the name "heart_disease_data". First the function `read_data()` checks, whether a dataset of this name is already registered in the workspace `ws` with `ws.datasets["heart_diseases_data"]`. If it is found, then the function will reuse this dataset.

If the dataset is not found, the function first accesses the default datastore of the workspace, and uploads the contents of the data folder into a folder named "data" in the datastore
```
datastore = ws.get_default_datastore()
datastore.upload('./data', overwrite=True, target_path='data')
```
Then a Tabular Dataset is created using `from_delimited_files()`. I could already register this dataset, but I want to use my own preprocessing algorithm to clean the data, therefore I call the function `prepare_data()` (see [section PreProcessing](#preprocessing)).
The output of this function is a cleaned and ready to use pandas dataframe. I register this dataframe as TabularData in my workspace with the name "heart_disease_data".
```
dataset = Dataset.Tabular.register_pandas_dataframe(df, datastore, show_progress=True,
                             name="heart_disease_data", description=description_text)
```

### PreProcessing
The preprocessing of the data is defined in the `prepare_data` function in the train.py script.
Some categorical features have to be preprocessed before they can be handled by the machine learning models.
- The columns Sex and ExerciseAngina are binary coded, with 1 for "male"/"yes" and 0 for "female"/"no" respectively.
- The column ST_Slope is numerically encoded with 1 for "Up", 0 for "Flat" and -1 for "Down".
- The columns for the ChestPainType and RestingECG features are one_hot encoded for the model.

## Automated ML
For the AutoML run I'm setting up a cluster with a vm size of `Standard_DS12_V2`, a maximum of 6 nodes and a minimum of 1 node.
```
compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_DS12_v2',
                                                          max_nodes=6, min_nodes=1)
compute_target = ComputeTarget.create(workspace=ws, name="expcluster", provisioning_configuration=compute_config)
```

The training task `classification` is run on the uploaded and cleaned dataset (see [section Dataset](#dataset)) with the target column `HeartDisease` on the above created cluster. I turned the featurization off, since I cleaned the data before registering the dataset. The results of the AutoML run are saved in the folder `./automl_run`.

```
automl_config = AutoMLConfig(compute_target=compute_target,
                             task="classification",
                             training_data=dataset,
                             label_column_name="HeartDisease",
                             path = './automl_run',
                             featurization = "off",
                             **automl_settings
                            )
```
The settings for the AutoML run are configured like this

```
automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "primary_metric": 'accuracy',
    "n_cross_validations": 5,
    "enable_early_stopping": True,
}
```
I chose the 20 min experiment time out, to ensure a completed run before the provided virtual machine times out. The AutoML run will use up to 5 cores in parallel (`max_concurrent_iterations`). I also enabled the early stopping policy, which will stop a run if there is no improvement after 31 iterations. The primary metric to evaluate the best model I chose `accuracy`. This is the most straightforward metric to judge a classification task. Since the target values are quite balanced in the dataset I don't expect much problems with the precision of the resulting model. Nevertheless, while evaluating the results of this run, I'm going to pay close attention to the confusion matrix and precison scores as well.
I'm using a 5-fold crossvalidation, since the dataset is very small and with crossvalidation the whole dataset is used to train the model. 

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

In the AutoML run tree based learning algorithms performed really well, especially the gradient boosted models XGBoostClassifiers and LightGBM. The worst performing models were based on Nearest-Neighbors-type algorithms (KNN, SVM). Most of the columns in my training data are binary, with only five features being numerical. 

KNN and SVM need a distance metric to work. AutoML chose the manhattan metric for the distance, which works good on standard normal and standard uniform distributions (--> numeric features) . But in this dataset the features will probably behave more like a Bernoulli distribution, where this metric simply fails, resulting in the low accuracy of these models.

On the other hand, many binary features mean that the distinction between leaves is easy for the algorithm. And with the balanced dataset, the resulting tree-models won't suffer much bias.

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
