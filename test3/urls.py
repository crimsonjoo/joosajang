from django.urls import path
from . import views

app_name = 'test3'

urlpatterns = [
    path('',views.index,name='index'),
    path('result',views.result,name='result'),
    path('result/detail',views.detail,name='detail'),
]