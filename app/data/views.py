from django.shortcuts import render
from rest_framework import status
from rest_framework import status
from io import BytesIO
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from django.conf import settings

from minio import Minio

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

MINIO_ACCESS_KEY = settings.MINIO_ACCESS_KEY
MINIO_SECRET_KEY = settings.MINIO_SECRET_KEY
BUCKET_NAME = settings.MINIO_BUCKET_NAME

minio_client = Minio(
    endpoint="localhost:9000",
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# Create your views here.
@api_view(['POST'])
def upload(request):
    if request.method == "POST":
        file = request.FILES.get('file')  # Access the uploaded file
        if file is not None:
            # Process the file (send it to Trino, save it, etc.)
            # Example:
            value_as_bytes = file.read()
            value_as_a_stream = BytesIO(value_as_bytes)
            minio_client.put_object("csv", "output.csv", data=value_as_a_stream, length=len(value_as_bytes))
            return Response("File uploaded successfully", status=status.HTTP_200_OK)
        else:
            return Response("No file uploaded", status=status.HTTP_400_BAD_REQUEST)
    else:
        return Response("Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED)


def read_data():
    try:
        df = pd.read_csv(f"s3://csv/output.csv",
                        encoding='unicode_escape',
                        storage_options={
                            "key": MINIO_ACCESS_KEY,
                            "secret": MINIO_SECRET_KEY,
                            "client_kwargs": {"endpoint_url": "http://localhost:9000/"}
                        })
        for i in df.columns:
            if "Unnamed" in i:
                df = df.drop(i, axis=1)
        return df
    except(Exception):
        print("File not found")
        return pd.DataFrame()

@api_view(['GET'])
def get_columns(request):
    if request.method == "GET":
        df = read_data()
        if not df.empty:
            metric_columns = []
            nominal_columns = []
            for i in df.columns:
                if is_numeric_dtype(df[i]):
                    metric_columns += [i]
                else:
                    nominal_columns += [i]
            respone = {"metric": metric_columns, "nominal": nominal_columns}
            return Response(respone, status=status.HTTP_202_ACCEPTED)
        else:
            return Response("File not found", status=status.HTTP_404_NOT_FOUND)
    else:
        return Response("Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['POST'])
def descriptive(request):
    if request.method == "POST":
        data = JSONParser().parse(request)
        metric = data['metric']
        ordinal = data['ordinal']
        method = data['method']
        df = read_data()
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
        return Response({"data": response_data}, status=status.HTTP_200_OK)
    else:
        return Response("Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED)

@api_view(["POST"])
def rfm(request):
    if request.method == "POST":
        df = read_data()
        if not df.empty:
            timestamp = request.POST.get("timestamp")
            monetary = request.POST.get("monetary")
            customer = request.POST.get("customer")
            try:
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
                response_data = []
                for i in range(len(rfm_df)):
                    for j in rfm_df.columns:
                        response_data += [{str(j): rfm_df.iloc[i][j]}]
                return Response({"data": rfm_df.head()}, status=status.HTTP_200_OK)
            except(Exception):
                print(Exception)
                return Response("Not valid timestamp column", status=status.HTTP_406_NOT_ACCEPTABLE)