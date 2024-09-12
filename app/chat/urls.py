from django.contrib import admin
from django.urls import path, include, re_path
from . import views
urlpatterns = [
    re_path(r'^$', views.chat ),
    path('delete/', views.delete_conversation),
    path('get-titles', views.get_title),
    path('get-data/', views.get_data)
]