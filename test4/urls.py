from django.urls import path
from . import views

app_name = 'test4'

urlpatterns = [
    path('',views.index,name='index'),
]