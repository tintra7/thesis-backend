from django.shortcuts import render
from rest_framework import status
from io import BytesIO
from rest_framework.response import Response
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.parsers import JSONParser
from django.conf import settings
from rest_framework import generics, authentication, permissions

from data.utils import (
    read_data,
    rfm_analysis,
    descriptive_analysis,
    mapping
)

from minio import Minio

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

MINIO_ENDPOINT = settings.MINIO_ENDPOINT
MINIO_ACCESS_KEY = settings.MINIO_ACCESS_KEY
MINIO_SECRET_KEY = settings.MINIO_SECRET_KEY
BUCKET_NAME = settings.MINIO_BUCKET_NAME


minio_client = Minio(
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# Create your views here.
@api_view(['POST'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def upload(request):
    if request.method == "POST":
        file = request.FILES.get('file')  # Access the uploaded file
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        if file is not None:
            print(type(file))
            value_as_bytes = file.read()
            value_as_a_stream = BytesIO(value_as_bytes)
            minio_client.put_object(BUCKET_NAME, file_name, data=value_as_a_stream, length=len(value_as_bytes))
            map = mapping([], [])
            response = {"message": "File uploaded successfully",
                        "mapping": map}
            return Response(response, status=status.HTTP_200_OK)
        else:
            return Response("No file uploaded", status=status.HTTP_400_BAD_REQUEST)
    else:
        return Response("Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['GET'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_columns(request):
    if request.method == "GET":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = read_data(file_name)
        if not df.empty:
            metric_columns = []
            nominal_columns = []
            for i in df.columns:
                if is_numeric_dtype(df[i]):
                    metric_columns += [i]
                else:
                    nominal_columns += [i]
            respone = {"metric": metric_columns, "nominal": nominal_columns}
            return Response(respone, status=status.HTTP_200_OK)
        else:
            return Response("File not found", status=status.HTTP_404_NOT_FOUND)
    else:
        return Response("Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED)

@api_view(['POST'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def descriptive(request):
    if request.method == "POST":
        data = JSONParser().parse(request)
        metric = data['metric']
        ordinal = data['ordinal']
        method = data['method']
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = read_data(file_name)
        response_data = descriptive_analysis(df, metric, ordinal, method)
        return Response({"data": response_data}, status=status.HTTP_200_OK)
    else:
        return Response("Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED)

@api_view(["POST"])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def rfm(request):
    if request.method == "POST":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = read_data(file_name)
        if not df.empty:
            timestamp = request.POST.get("timestamp")
            monetary = request.POST.get("monetary")
            customer = request.POST.get("customer")
            try:
                rfm_df = rfm_analysis(df, timestamp, monetary, customer)
                response_data = []
                for i in range(len(rfm_df)):
                    for j in rfm_df.columns:
                        response_data += [{str(j): rfm_df.iloc[i][j]}]
                return Response({"data": rfm_df.head()}, status=status.HTTP_200_OK)
            except(Exception):
                print(Exception)
                return Response("Not valid timestamp column", status=status.HTTP_406_NOT_ACCEPTABLE)
        else:
            return Response("File not found", status=status.HTTP_404_NOT_FOUND)