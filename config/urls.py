"""config URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import include, path
from django.contrib import admin
from django.conf.urls import url,handler400,handler500,handler404

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('main.urls')),
    path('test1/', include('test1.urls')),
    path('test2/', include('test2.urls')),
    path('test3/', include('test3.urls')),
    path('test4/', include('test4.urls')),
]


handler400 = 'main.views.error400'
handler404 = 'main.views.error404'
handler500 = 'main.views.error500'
