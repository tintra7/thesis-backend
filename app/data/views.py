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
import json
import time
import os
from data.utils import (
    MinioClient,
    rfm_analysis,
    descriptive_analysis,
    get_mapping,
    data_preprocessing,
    train_with_lstm,
    train_with_prophet,
    train_with_xgboost,
    try_parse_datetime,
    filter_data,
    create_sql_engine,
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

@api_view(['GET', 'DELETE', 'POST'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def data(request):
    if request.method == "GET":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = minio_client.read_csv(file_name)

        if not df.empty:
            # paginator = CustomPagination()
            # result_page = paginator.paginate_queryset(df.to_dict('records'), request)
            # response_data = result_page
            # return paginator.get_paginated_response(response_data)
            df = df.dropna(axis=0)
            return Response({"data": df.to_dict('records')}, status=status.HTTP_200_OK)
        else:
            return Response({"message":"File not found"}, status=status.HTTP_404_NOT_FOUND)
    if request.method == "DELETE":
        user_id = request.user.id
        remove_status = minio_client._remove_object(user_id=user_id)
        # Delete sqllite file
        if os.path.exists(f"app/database/{user_id}.db"):
            os.remove(f"app/database/{user_id}.db")
        else:
            print("The file does not exist")
        if remove_status:
            return Response(f"Clear all file of {user_id}", status=status.HTTP_200_OK)
        
        else:
            return Response("Nothing to delete", status=status.HTTP_200_OK)
    if request.method == "POST":
        file = request.FILES.get('file')  # Access the uploaded file
        file_type = request.data.get("type")
        delimiter = request.data.get("delimiter")
        user_id = request.user.id
        current_time = int(time.time())
        file_name = f"{user_id}/file.csv"
        current_file_name =f"{user_id}/file_{current_time}.csv"
        if file is not None:
            value_as_bytes = file.read()
            df = pd.read_csv(StringIO(value_as_bytes.decode('utf-8')))
            if not minio_client._exist_file(user_id, "mapping.json"):
                map = get_mapping(list(df.columns))
            else:
                map = minio_client.read_json(f"{user_id}/mapping.json")
                df = df.rename(map, axis=1)
            # if exist target file, concat newfile to target file
            if minio_client._exist_file(user_id, "file.csv"):
                target_df = minio_client.read_csv(f"{user_id}/file.csv")
                target_df = pd.concat([target_df, df]).drop_duplicates()
                print(target_df.head())
                is_uploaded = minio_client.to_csv(target_df, file_name)
            else:
                is_uploaded = minio_client.to_csv(df, file_name)
            if not is_uploaded:
                return Response({"message" : "Uploadfile failed"}, status=status.HTTP_400_BAD_REQUEST)  
            is_uploaded = minio_client.to_csv(df, current_file_name)
            if not is_uploaded:
                return Response({"message" : "Uploadfile failed"}, status=status.HTTP_400_BAD_REQUEST)
            
            response = {"message": "File uploaded successfully",
                        "mapping": map}
            return Response(response, status=status.HTTP_200_OK)
        else:
            return Response("No file uploaded", status=status.HTTP_400_BAD_REQUEST)
    else:
        return Response({"message":"Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

@api_view(['GET'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_columns(request):
    if request.method == "GET":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = minio_client.read_csv(file_name)
        if not df.empty:
            columns = [i for i in df.columns]
            response = {"columns": columns}
            return Response(response, status=status.HTTP_200_OK)
        else:
            return Response({"message":"File not found"}, status=status.HTTP_404_NOT_FOUND)
        # if not df.empty:
        #     metric_columns = []
        #     nominal_columns = []
        #     for i in df.columns:
        #         if is_numeric_dtype(df[i]):
        #             metric_columns += [i]
        #         else:
        #             nominal_columns += [i]
        #     response = {"metric": metric_columns, "nominal": nominal_columns}
        #     return Response(response, status=status.HTTP_200_OK)
        # else:
        #     return Response({"message":"File not found"}, status=status.HTTP_404_NOT_FOUND)
    else:
        return Response({"message":"Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

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
        return Response({"message":"Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

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
            customer = "Customer Name" if "Customer Name" in df.columns else "Customer ID"
            if timestamp not in df.columns:
                return Response({"message": "Missing datetime colunm"}, status=status.HTTP_400_BAD_REQUEST)
            if monetary not in df.columns:
                return Response({"message": "Missing monetary colunm"}, status=status.HTTP_400_BAD_REQUEST)
            if customer not in df.columns:
                return Response({"message": "Missing customer colunm"}, status=status.HTTP_400_BAD_REQUEST)
            try:
                rfm_df = rfm_analysis(df, timestamp, monetary, customer)
                response_data = rfm_df.to_dict('records')
                counts = rfm_df['Customer_segment'].value_counts().to_dict()
                return Response({"data": response_data, 'value_counts': counts, 'total': len(rfm_df)}, status=status.HTTP_200_OK)
            except(Exception):
                return Response({"message": "Calculate RFM failed"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response({"message":"File not found"}, status=status.HTTP_404_NOT_FOUND)
    else:
        return Response({"message":"Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


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
            return Response({"message":"File not found"}, status=status.HTTP_404_NOT_FOUND)
    else:
        return Response({"message":"Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


@api_view(['GET', 'PUT', 'POST'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def mapping(request):
    if request.method == "GET":
        user_id = request.user.id
        file_name = f"{user_id}/mapping.json"
        dic = minio_client.read_json(file_name)
        if dic:
            return Response({"mapping": dic}, status=status.HTTP_200_OK)
        else:
            return Response({"message": "File not found"}, status=status.HTTP_404_NOT_FOUND)
    if request.method == "POST":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = minio_client.read_csv(file_name)
        file_name = f"{user_id}/mapping.json"
        dic = request.data.get("mapping", "")

        if dic == "":
            return Response({"message":"Missing mapping"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            dic = json.loads(dic)
        except:
            return Response({"message":"JSON format is not compatible"}, status=status.HTTP_406_NOT_ACCEPTABLE)
        if "Date" not in list(dic.values()):
            return Response({"message":"Your data must have a datetime column"}, status=status.HTTP_406_NOT_ACCEPTABLE)
        is_load = minio_client.to_json(dic, file_name=file_name)
        if is_load:
            df = df.rename(dic, axis=1)
            df = data_preprocessing(df)
            datetime = try_parse_datetime(df['Date'])
            if not datetime.empty:
                df['Date'] = datetime
            minio_client.to_csv(df, file_name=f"{user_id}/file.csv")
            engine = create_sql_engine(user_id)
            try:
                df.to_sql("data", engine, index=False)
            except:
                return Response({"message" : "Create sqlite engine fail"}, status=status.HTTP_400_BAD_REQUEST)
            return Response({"message":"Upload successfuly"}, status=status.HTTP_200_OK)
        else:
            return Response({"message":"Update mapping failed"}, status=status.HTTP_400_BAD_REQUEST)
        
    if request.method == "PUT":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = minio_client.read_csv(file_name)
        file_name = f"{user_id}/mapping.json"
        dic = request.data.get("mapping", "")
        if dic == "":
            return Response({"message":"Missing mapping"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            dic = json.loads(dic)
            
        except:
            return Response({"message":"JSON format is not compatible"}, status=status.HTTP_406_NOT_ACCEPTABLE)
        if "Date" not in list(dic.values()):
            return Response({"message":"Your data must have a datetime column"}, status=status.HTTP_406_NOT_ACCEPTABLE)
        is_load = minio_client.to_json(dic, file_name=file_name)
        
        if is_load:
            df = df.rename(dic, axis=1)
            datetime = try_parse_datetime(df['Date'])
            if not datetime.empty:
                df['Date'] = datetime
            minio_client.to_csv(df, file_name=f"{user_id}/file.csv")
            return Response({"message":"Update successfuly","mapping": dic}, status=status.HTTP_200_OK)
        else:
            return Response({"message":"Update mapping failed"}, status=status.HTTP_400_BAD_REQUEST)

    else:
        return Response({"message":"Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)
        
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
        filters = request.GET.get('filter', "")
        if filters != "":
            try:
                filters = json.loads(filters)
                try:
                    df = filter_data(df, filters)
                except:
                    return Response({"message": "Apply filter failed"}, status=status.HTTP_400_BAD_REQUEST)
            except:
                return Response({"message": "Filter format not excepted"}, status=status.HTTP_400_BAD_REQUEST)
        if not df.empty:
            if "Date" not in df.columns:
                return Response({"message": "Date column not found"}, status=status.HTTP_404_NOT_FOUND)
            test_size = 0.8
            prophet_model = train_with_prophet(df, test_size, target)
            xgboost_model = train_with_xgboost(df, test_size, target, time_range=time_range)
            lstm_model = train_with_lstm(df, test_size, target, time_range=time_range)
            
            if prophet_model.eval.mse() < min(xgboost_model.eval.mse(), lstm_model.eval.mse()):
                final_model = prophet_model
            elif xgboost_model.eval.mse() < lstm_model.eval.mse():
                final_model = xgboost_model
            else:
                final_model = lstm_model

            response = {
                "model_name": final_model.name,
                "mse": final_model.eval.mse(), 
                "mae": final_model.eval.mae(), 
                "value": final_model._make_predict(time_range), 
                "eval": final_model.eval.get_eval_df()
                }
            return Response(response, status=status.HTTP_200_OK)
        else:
            return Response({"message":"File not found"}, status=status.HTTP_404_NOT_FOUND)
    else:
        return Response({"message":"Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)