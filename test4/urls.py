from django.urls import path
from . import views

app_name = 'test4'

urlpatterns = [
    path('',views.index,name='index'),
    path('question/',views.question,name='question'),
    path('predict/',views.predict,name='predict'),
    path('predict/<int:num>',views.result_predict,name='result_predict'),
    path('classify/',views.classify,name='classify'),
    path('classify/<int:num>',views.result_classify,name='result_classify'),
    path('detail/predict/<int:idx>',views.detail_predict,name='detail_predict'),
    path('detail/classify/<int:idx>',views.detail_classify,name='detail_classify'),
]