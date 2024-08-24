from django.urls import re_path, path
from chart import views

urlpatterns = [
    re_path(r'^value/$',views.get_value),
    re_path(r'^histplot/$', views.get_histplot),
    re_path(r'^linechart/$', views.get_linechart),
    re_path(r'^barchart/$', views.get_barchart),
    re_path(r'^piechart/$', views.get_piechart),
]