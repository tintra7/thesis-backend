from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework import generics, authentication, permissions
from chart.utils import (
    read_data,
    calculate_value,
    validate_function
)
from rest_framework import status
from rest_framework.response import Response
import numpy as np

@api_view(['GET'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def get_value(request):
    if request.method == "GET":
        user_id = request.user.id
        file_name = f"{user_id}/file.csv"
        df = read_data(file_name)
        function = request.get("function", "")
        column = request.get("column", "")
        if function == "":
            return Response("Function not found", status=status.HTTP_404_NOT_FOUND)
        if column == "":
            return Response("Column not found", status=status.HTTP_404_NOT_FOUND)
        if not df.empty:
            if not validate_function(function):
                return Response("Function not accepted", status=status.HTTP_404_NOT_FOUND)
            res = calculate_value(df, function, column)
            return Response({"value": int(res)}, status=status.HTTP_200_OK)
        else:
            return Response("File not found", status=status.HTTP_404_NOT_FOUND)