from io import BytesIO
from minio import Minio

from django.conf import settings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

MINIO_ACCESS_KEY = settings.MINIO_ACCESS_KEY
MINIO_SECRET_KEY = settings.MINIO_SECRET_KEY
BUCKET_NAME = settings.MINIO_BUCKET_NAME

minio_client = Minio(
    endpoint="minio:9000",
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

def read_data():
    try:
        if not minio_client.bucket_exists(BUCKET_NAME):
            print(f"Bucket {BUCKET_NAME} does not exist.")
            return pd.DataFrame()
    except:
        print("Bucket not found")

    try:
        file_name = "output.csv"
        df = pd.read_csv(f"s3://{BUCKET_NAME}/{file_name}",
                        encoding='unicode_escape',
                        storage_options={
                            "key": MINIO_ACCESS_KEY,
                            "secret": MINIO_SECRET_KEY,
                            "client_kwargs": {"endpoint_url": "http://{MINIO_ENDPOINT}"}
                        })
        for i in df.columns:
            if "Unnamed" in i:
                df = df.drop(i, axis=1)
        return df
    except(Exception):
        print("File not found")
        return pd.DataFrame()

def rfm_analysis(df, timestamp, monetary, customer):
    df[timestamp] = pd.to_datetime(df[timestamp])
    df_recency = df.groupby(by=customer,
            as_index=False)[timestamp].max()
    df_recency.columns = [customer, 'LastPurchaseDate']
    recent_date = df_recency['LastPurchaseDate'].max()
    df_recency['Recency'] = df_recency['LastPurchaseDate'].apply(
        lambda x: (recent_date - x).days)

    frequency_df = df.drop_duplicates().groupby(
        by=[customer], as_index=False)[timestamp].count()
    frequency_df.columns = [customer, 'Frequency']
    monetary_df = df.groupby(by=customer, as_index=False)[monetary].sum()
    monetary_df.columns = [customer, 'Monetary']

    rf_df = df_recency.merge(frequency_df, on=customer)
    rfm_df = rf_df.merge(monetary_df, on=customer).drop(
        columns='LastPurchaseDate')
    rfm_df['R_rank'] = rfm_df['Recency'].rank(ascending=False)
    rfm_df['F_rank'] = rfm_df['Frequency'].rank(ascending=True)
    rfm_df['M_rank'] = rfm_df['Monetary'].rank(ascending=True)

    # normalizing the rank of the customers
    rfm_df['R_rank_norm'] = (rfm_df['R_rank']/rfm_df['R_rank'].max())*100
    rfm_df['F_rank_norm'] = (rfm_df['F_rank']/rfm_df['F_rank'].max())*100
    rfm_df['M_rank_norm'] = (rfm_df['F_rank']/rfm_df['M_rank'].max())*100
    rfm_df.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

    rfm_df['RFM_Score'] = 0.15*rfm_df['R_rank_norm']+0.28 * \
        rfm_df['F_rank_norm']+0.57*rfm_df['M_rank_norm']
    rfm_df['RFM_Score'] *= 0.05
    rfm_df = rfm_df.round(2)

    rfm_df["Customer_segment"] = np.where(rfm_df['RFM_Score'] >
                                        4.5, "Top Customers",
                                        (np.where(
                                            rfm_df['RFM_Score'] > 4,
                                            "High value Customer",
                                            (np.where(
        rfm_df['RFM_Score'] > 3,
                                "Medium Value Customer",
                                np.where(rfm_df['RFM_Score'] > 1.6,
                                'Low Value Customers', 'Lost Customers'))))))

    return rfm_df

def descriptive_analysis(df, metric, ordinal, method):
    response_data = []
    if ordinal:
        method += [i for i in ordinal]
        df = df.groupby(ordinal)[metric].describe().reset_index()
        print(df.shape)
        for i in range(len(df.index)):
            for j in method:
                response_data += [{str(j): df.iloc[i][j]}]
    else:
        df = df[metric].describe().reset_index()
        for i in method:
            data = df.loc[df['index'] == i]
            response_data += [{str(data["index"].values[0]) : float(data[metric])}]
    return response_data