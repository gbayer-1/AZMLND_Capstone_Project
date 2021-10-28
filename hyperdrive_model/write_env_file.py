# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 09:44:11 2021

@author: GABRIFR

write the environment yml for the deployment of the onnx model
"""

from azureml.core.conda_dependencies import CondaDependencies

myenv = CondaDependencies.create(pip_packages=["numpy",
                                               "pandas",
                                               "onnxruntime",
                                               "azureml-core",
                                               "azureml-defaults"])

with open("hyperdrive_env.yml","w") as f:
    f.write(myenv.serialize_to_string())