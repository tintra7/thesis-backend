from django.contrib import admin
from django.urls import path, include, re_path
from . import views
urlpatterns = [
    re_path(r'^$', views.chat ),
    path('get-titles', views.get_title),
    path('get-data/', views.get_data)
]