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
    global session, input_name, output_name
    model = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'hyperdrive_model.onnx')
    session = onnxruntime.InferenceSession(model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
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
method_sample = StandardPythonParameterType("predict")
@input_schema("method", method_sample, convert_to_provided_type=False)
@input_schema("data", PandasParameterType(input_sample), convert_to_provided_type=False)
@output_schema(NumpyParameterType(output_sample))

def preprocess(input_data_json):
    dat = json.loads(input_data_json)['data']
    dummy_df = pd.DataFrame(json.loads(dat))
    return np.array(dummy_df.astype(np.float32))

def run(input_data):
    try:
        dat = preprocess(input_data)
        result = session.run([output_name], {input_name: dat})
        return json.dumps({"result": result[0].tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})