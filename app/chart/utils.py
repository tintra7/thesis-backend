from minio import Minio

from django.conf import settings

import numpy as np
import pandas as pd

MINIO_ACCESS_KEY = settings.MINIO_ACCESS_KEY
MINIO_SECRET_KEY = settings.MINIO_SECRET_KEY
BUCKET_NAME = settings.MINIO_BUCKET_NAME
MINIO_ENDPOINT = settings.MINIO_ENDPOINT
minio_client = Minio(
    endpoint="minio:9000",
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

def read_data(file_name):
    try:
        if not minio_client.bucket_exists(BUCKET_NAME):
            print(f"Bucket {BUCKET_NAME} does not exist.")
            return pd.DataFrame()
    except:
        print("Bucket not found")

    try:
        df = pd.read_csv(f"s3://{BUCKET_NAME}/{file_name}",
                        encoding='unicode_escape',
                        storage_options={
                            "key": MINIO_ACCESS_KEY,
                            "secret": MINIO_SECRET_KEY,
                            "client_kwargs": {"endpoint_url": f"http://{MINIO_ENDPOINT}"}
                        })
        for i in df.columns:
            if "Unnamed" in i:
                df = df.drop(i, axis=1)
        return df
    except(Exception):
        print(Exception)
        return pd.DataFrame()

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

def create_pivot(data, values, index, aggfunc, columns):
    pivot_table = pd.pivot(data=data, values=values, index=index, aggfunc=aggfunc, columns=columns)
