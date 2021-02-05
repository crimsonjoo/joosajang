from django.urls import path
from . import views

app_name = 'test1'

urlpatterns = [
    path('',views.index,name='index'),
    path('index2/',views.index2,name='index2'),
    path('question1/',views.question1,name='question1'),
    path('question2/',views.question2,name='question2'),
    path('question3/',views.question3,name='question3'),
    path('question4/0',views.question4_0,name='question4_0'),
    path('question4/1',views.question4_1,name='question4_1'),
    path('result/<int:num>',views.result,name='result'),
    path('result/detail/<int:idx>',views.detail,name='detail'),
]