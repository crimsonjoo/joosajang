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
from django.urls import path
from django.contrib import admin
from django.conf.urls import include
from django.http import HttpResponse

# sitemap import
from .sitemaps import *
from django.contrib.sitemaps.views import sitemap


sitemaps = {
    'static':StaticViewSitemap,
}


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('main.urls')),
    path('test1/', include('test1.urls')),
    path('test2/', include('test2.urls')),
    path('test3/', include('test3.urls')),
    path('test4/', include('test4.urls')),
    path('robots.txt/', lambda x: HttpResponse("User-agent: *Allow:/", content_type="text/plain")),
    path('sitemap.xml',sitemap,{'sitemaps':sitemaps},name='sitemap'),
]


handler400 = 'main.views.error400'
handler404 = 'main.views.error404'
handler500 = 'main.views.error500'
