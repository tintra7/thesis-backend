from minio import Minio

from django.conf import settings

import numpy as np
import pandas as pd

MINIO_ACCESS_KEY = settings.MINIO_ACCESS_KEY
MINIO_SECRET_KEY = settings.MINIO_SECRET_KEY
BUCKET_NAME = settings.MINIO_BUCKET_NAME
MINIO_ENDPOINT = settings.MINIO_ENDPOINT


def validate_function(function):
    function_list = {"sum", "mean", "min", "max", "std", "median", "count"}
    if function not in function_list:
        return False
    return True

def calculate_value(df, function, column):
    if function == "sum":
        res = np.sum(df[column])
    elif function == "mean":
        res = np.mean(df[column])
    elif function == "max":
        res = np.max(df[column])
    elif function == "min":
        res = np.min(df[column])
    elif function == "std":
        res = np.std(df[column])
    elif function == "median":
        res = np.median(df[column])
    elif function == "count":
        res = len(df[column].unique())
    return res

def create_pivot(data, values, index, aggfunc, column):
    res = []
    if column != "":
        pivot_table = pd.pivot_table(data=data, values=values, index=index, aggfunc=aggfunc, columns=column, fill_value=0)
        for col in pivot_table.columns:
            res.append({
                "x": list(pivot_table.index),
                "y": list(pivot_table[col].values),
                'name': col
            })
    else:
        pivot_table = pd.pivot_table(data=data, values=values, index=index, aggfunc=aggfunc, fill_value=0)

        res = [{"x": list(pivot_table.index),
                "y": list(pivot_table.values.flatten())}]
    return res
