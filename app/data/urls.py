from django.urls import re_path, path
from data import views

urlpatterns = [

    re_path(r'^load/$',views.upload),
    re_path(r'^data-column/$', views.get_columns),
    re_path(r'^descriptive-analysis/$', views.descriptive),
    re_path(r'^rfm/$', views.rfm),
]