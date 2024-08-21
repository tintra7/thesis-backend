from rest_framework import status
from io import BytesIO
from rest_framework.response import Response
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.parsers import JSONParser
from django.conf import settings
from rest_framework import generics, authentication, permissions
from rest_framework.pagination import PageNumberPagination
from io import StringIO
from prophet import Prophet

from data.utils import (
    MinioClient,
    rfm_analysis,
    descriptive_analysis,
    mapping,
    data_preprocessing,
    train_with_prophet
)

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

MINIO_ENDPOINT = settings.MINIO_ENDPOINT
MINIO_ACCESS_KEY = settings.MINIO_ACCESS_KEY
MINIO_SECRET_KEY = settings.MINIO_SECRET_KEY
BUCKET_NAME = settings.MINIO_BUCKET_NAME
MAX_PAGE_SIZE = settings.MAX_PAGE_SIZE


minio_client = MinioClient(
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

class CustomPagination(PageNumberPagination):
    page_size_query_param = 'page_size'
    max_page_size = MAX_PAGE_SIZE

# Create your views here.
@api_view(['POST'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def upload(request):
    if request.method == "POST":
        file = request.FILES.get('file')  # Access the uploaded file
        file_type = request.POST.get("type")
        delimiter = request.POST.get("delimiter")
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"

        if file is not None:
            value_as_bytes = file.read()
            df = pd.read_csv(StringIO(value_as_bytes.decode('utf-8')))
            map = mapping(df.columns, [])
            df = data_preprocessing(df.rename(map, axis=1))
            is_uploaded = minio_client.to_csv(df, file_name)
            if not is_uploaded:
                return Response({"message" : "Uploadfile failed"}, status=status.HTTP_400_BAD_REQUEST)
            is_uploaded = minio_client.to_json(map, f"{user_id}/mapping.json")
            if not is_uploaded:
                return Response({"message" : "Uploadfile failed"}, status=status.HTTP_400_BAD_REQUEST)
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
        df = minio_client.read_csv(file_name)
        if not df.empty:
            metric_columns = []
            nominal_columns = []
            for i in df.columns:
                if is_numeric_dtype(df[i]):
                    metric_columns += [i]
                else:
                    nominal_columns += [i]
            response = {"metric": metric_columns, "nominal": nominal_columns}
            return Response(response, status=status.HTTP_200_OK)
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
        df = minio_client.read_csv(file_name)
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
        mapper_name = f"{user_id}/mapping.json"
        df = minio_client.read_csv(file_name)
        mapper = minio_client.read_json(file_name=mapper_name)
        if not df.empty and mapper:
            df = df.rename(mapper, axis=1)
            timestamp = "Date"
            monetary = "Total Price"
            customer = "Custormer ID"
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
    else:
        return Response("Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['GET'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_data_length(request):
    if request.method == "GET":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = minio_client.read_csv(file_name)
        if not df.empty:
            response = {"length": len(df)}
            return Response(response, status=status.HTTP_200_OK)
        else:
            return Response("File not found", status=status.HTTP_404_NOT_FOUND)
    else:
        return Response("Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED)

@api_view(['GET', 'DELETE'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def data(request):
    if request.method == "GET":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = minio_client.read_csv(file_name)

        if not df.empty:
            paginator = CustomPagination()
            result_page = paginator.paginate_queryset(df.to_dict('records'), request)
            response_data = result_page
            return paginator.get_paginated_response(response_data)
        else:
            return Response("File not found", status=status.HTTP_404_NOT_FOUND)
    if request.method == "DELETE":
        user_id = request.user.id
        remove_status = minio_client._remove_object(user_id=user_id)
        if remove_status:
            return Response(f"Clear all file of {user_id}", status=status.HTTP_200_OK)
        else:
            return Response("Nothing to delete", status=status.HTTP_200_OK)
    else:
        return Response("Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED)

@api_view(['GET'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_mapping(request):
    if request.method == "GET":
        user_id = request.user.id
        file_name = f"{user_id}/mapping.json"
        dic = minio_client.read_json(file_name)
        if dic:
            return Response({"mapping": dic}, status=status.HTTP_200_OK)
        else:
            return Response("File not found", status=status.HTTP_404_NOT_FOUND)
    
    else:
        return Response("Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED)
        
@api_view(['GET'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def forecast(request):
    if request.method == "GET":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = minio_client.read_csv(file_name)
        time_range = int(request.GET.get('time'))
        target = request.GET.get("metric")
        if not df.empty:
            if "Date" not in df.columns:
                return Response("Date column not found", status=status.HTTP_404_NOT_FOUND)
            test_size = 0.8
            model = train_with_prophet(df, test_size, target)
            future_sum = model._make_predict(time_range)
            response = {
                "mse": model.eval.mse(), 
                "mae": model.eval.mae(), 
                "value": future_sum, 
                "eval": model.eval.get_eval_df().to_dict('records')
                }

            return Response(response, status=status.HTTP_200_OK)
        else:
            return Response("File not found", status=status.HTTP_404_NOT_FOUND)
    else:
        return Response("Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED)