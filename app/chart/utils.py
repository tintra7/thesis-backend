from minio import Minio

from django.conf import settings

import numpy as np
import pandas as pd


MINIO_ACCESS_KEY = settings.MINIO_ACCESS_KEY
MINIO_SECRET_KEY = settings.MINIO_SECRET_KEY
BUCKET_NAME = settings.MINIO_BUCKET_NAME
MINIO_ENDPOINT = settings.MINIO_ENDPOINT


def contain_columns(column, data_columns):
    return column in data_columns

def validate_function(function):
    function_list = {"sum", "mean", "min", "max", "std", "median", "count", "unique"}
    if function not in function_list:
        return False
    return True

def validate_filter(filter_list):
    if filter_list == []:
        return True
    for filter in filter_list:
        if filter['type'] == 'daterange':
            from_date = pd.to_datetime(filter['from_to'][0])
            to_date = pd.to_datetime(filter['from_to'][1])
            if from_date > to_date:
                return False
        elif filter['type'] == 'range':
            from_value = pd.to_numeric(filter['from_to'][0])
            to_value = pd.to_numeric(filter['from_to'][1])
            if from_value > to_value:
                return False
    return True

def get_filter(df, filter_list):
    if filter_list == []:
        return df
    filter_df = df.copy()
    for filter in filter_list:
        column = filter['column']
        type = filter['type']
        if type == "include":
            filter_df = filter_df[filter_df[column].isin(filter['values'])]
        elif type == "daterange":
            from_date = pd.to_datetime(filter['from_to'][0])
            to_date = pd.to_datetime(filter['from_to'][1])
            filter_df[column] = pd.to_datetime(filter_df[column])
            filter_df = filter_df[(filter_df[column] >= from_date) & (filter_df[column] <= to_date)]
        elif type == "range":
            from_value = pd.to_numeric(filter['from_to'][0])
            to_value = pd.to_numeric(filter['from_to'][1])
            filter_df = filter_df[(filter_df[column] >= from_value) & (filter_df[column] <= to_value)]
    return filter_df
                

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
    elif function == "unique":
        res = df[column].unique()
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

def create_boxplot(data: pd.DataFrame, x: str, y: str):
    res = {}
    res['data'] = []
    unique_values = data[x].unique()
    for value in unique_values:
        # res['cols'].append(value)
        res['data'].append({'y': list(data[data[x] == value][y].values), 'name': value})
        # res[value] = list(data[data[x] == value][y].values)
    return res