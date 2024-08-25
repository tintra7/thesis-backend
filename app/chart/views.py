from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework import generics, authentication, permissions
from chart.utils import (
    calculate_value,
    validate_function,
    create_pivot,
)

from data.utils import (
    MinioClient
)
from rest_framework import status
from rest_framework.response import Response
import numpy as np
from pandas.api.types import is_numeric_dtype
from django.conf import settings


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

@api_view(['GET'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_value(request):
    if request.method == "GET":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = minio_client.read_csv(file_name)
        function = request.GET.get("function", "")
        column = request.GET.get("column", "")
        if function == "":
            return Response({"message": "Function not found"}, status=status.HTTP_404_NOT_FOUND)
        if not validate_function(function):
            return Response({"message": "Function not accepted"}, status=status.HTTP_404_NOT_FOUND)
        if column == "" or column not in df.columns:
            return Response({"message":"Column not found"}, status=status.HTTP_404_NOT_FOUND)

        if not df.empty:
            
            res = calculate_value(df, function, column)
            return Response({"value": int(res)}, status=status.HTTP_200_OK)
        else:
            return Response({"message":"File not found"}, status=status.HTTP_404_NOT_FOUND)
    else:
        return Response({"message:": "Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)
    
@api_view(['GET'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_linechart(request):
    if request.method == "GET":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = minio_client.read_csv(file_name)
        column = request.GET.get("column", "")
        function = request.GET.get("function", "")
        x = request.GET.get("x", "")
        y = request.GET.get("y", "")
        if x == "":
            return Response({"message":"X axis not found"}, status=status.HTTP_404_NOT_FOUND)
        if y == "":
            return Response({"message":"Y axis not found"}, status=status.HTTP_404_NOT_FOUND)
        if function == "":
            return Response({"message": "Function not found"}, status=status.HTTP_404_NOT_FOUND)
        if not validate_function(function):
            return Response({"message": "Function not accepted"}, status=status.HTTP_404_NOT_FOUND)
        if not df.empty:
            print(df.columns)
            res = create_pivot(data=df, values=y, index=x, aggfunc=function, column=column)
            
            return Response({"data": res}, status=status.HTTP_200_OK)
        else:
            return Response({"message":"File not found"}, status=status.HTTP_404_NOT_FOUND)
    else:
        return Response({"message:": "Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)
    
@api_view(['GET'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_barchart(request):
    if request.method == "GET":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = minio_client.read_csv(file_name)
        column = request.GET.get("column", "")
        function = request.GET.get("function", "")
        x = request.GET.get("x", "")
        y = request.GET.get("y", "")
        if x == "":
            return Response({"message":"X axis not found"}, status=status.HTTP_404_NOT_FOUND)
        if y == "":
            return Response({"message":"Y axis not found"}, status=status.HTTP_404_NOT_FOUND)
        if function == "":
            return Response({"message": "Function not found"}, status=status.HTTP_404_NOT_FOUND)
        if not validate_function(function):
            return Response({"message": "Function not accepted"}, status=status.HTTP_404_NOT_FOUND)
        if not df.empty:
            print(df.columns)
            res = create_pivot(data=df, values=y, index=x, aggfunc=function, column=column)
            return Response({"data": res}, status=status.HTTP_200_OK)
        else:
            return Response({"message":"File not found"}, status=status.HTTP_404_NOT_FOUND)
    else:
        return Response({"message:": "Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)
    

@api_view(['GET'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_histplot(request):
    if request.method == "GET":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = minio_client.read_csv(file_name)
        column = request.GET.get("column", "")
        if column == "" or column not in df.columns:
            return Response("Column not found", status=status.HTTP_404_NOT_FOUND)
        if df.empty:
            return Response({"message":"File not found"}, status=status.HTTP_404_NOT_FOUND)
        res = {"data": list(df[column].values)}
        return Response(res, status=status.HTTP_200_OK)
    else:
        return Response({"message:": "Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)
    
@api_view(['GET'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_piechart(request):
    if request.method == "GET":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = minio_client.read_csv(file_name)
        labels = request.GET.get("labels", "")
        values = request.GET.get("values", "")
        function = request.GET.get("function", "")
        if function == "":
            return Response({"message": "Function not found"}, status=status.HTTP_404_NOT_FOUND)
        if not validate_function(function):
            return Response({"message": "Function not accepted"}, status=status.HTTP_404_NOT_FOUND)
        if labels == "" or labels not in df.columns or values == "" or values not in df.columns:
            return Response("Column not found", status=status.HTTP_404_NOT_FOUND)
        try:
            pivot_table = create_pivot(data=df, values=values, index=labels, aggfunc=function, column="")
            print(pivot_table[0]['y'])
            values_data = [i[0] for i in pivot_table[0]['y']]
            res = {"label": pivot_table[0]["x"], "values": values_data}
            return Response(res, status=status.HTTP_200_OK)
        except Exception:
            return Response({"message":"Function not allowed"}, status=status.HTTP_406_NOT_ACCEPTABLE)

    else:
        return Response({"message:": "Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

