# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 08:40:42 2021

@author: GABRIFR
"""
import os
import onnxruntime
import pandas as pd
import numpy as np
import json

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

def init():
    '''
    init function is called to load the model.
    I invoke a onnxruntime.Inferencesession
    '''
    global session, input_name, output_name
    model = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'hyperdrive_model.onnx')
    session = onnxruntime.InferenceSession(model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

# define sample input and output to be used to create a swagger file.
input_sample = pd.DataFrame({"Age": pd.Series([0], dtype="int64"),
                             "Sex": pd.Series([0], dtype="int64"),
                             "RestingBP": pd.Series([0], dtype="int64"),
                             "Cholesterol": pd.Series([0], dtype="int64"),
                             "FastingBS": pd.Series([0], dtype="int64"),
                             "MaxHR": pd.Series([0], dtype="int64"),
                             "ExerciseAngina": pd.Series([0], dtype="int64"),
                             "Oldpeak": pd.Series([0.0], dtype="float64"),
                             "ST_Slope": pd.Series([0], dtype="int64"),
                             "ChestPainType_ASY": pd.Series([0], dtype="int64"),
                             "ChestPainType_ATA": pd.Series([0], dtype="int64"),
                             "ChestPainType_NAP": pd.Series([0], dtype="int64"),
                             "ChestPainType_TA": pd.Series([0], dtype="int64"),
                             "RestingECG_LVH": pd.Series([0], dtype="int64"),
                             "RestingECG_Normal": pd.Series([0], dtype="int64"),
                             "RestingECG_ST": pd.Series([0], dtype="int64")})
output_sample = np.array([0])

sample_input = StandardPythonParameterType([PandasParameterType(input_sample)])
@input_schema("Inputs", sample_input)
@output_schema(StandardPythonParameterType({"Results": NumpyParameterType(output_sample)}))

def run(Inputs):
    '''
    run function is used to run the model.
    the input is processed to be fed to the model and the result is returned.
    '''
    try:
        # reshape the pandas dataframe to the FloatTensorType Input the Onnx models expects.
        data = np.array(Inputs[0].astype(np.float32))
        result = session.run([output_name], {input_name: data})
        return result[0].tolist()
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})