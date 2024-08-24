from django.urls import re_path, path
from data import views

urlpatterns = [

    re_path(r'^$', views.data),
    re_path(r'^data-column/$', views.get_columns),
    re_path(r'^descriptive-analysis/$', views.descriptive),
    re_path(r'^rfm/$', views.rfm),
    re_path(r'^row/$', views.get_data_length),
    re_path(r'^forecast/$', views.forecast),
    re_path(r'^mapping/$', views.get_mapping),

]