from django.urls import reverse
from django.contrib.sitemaps import Sitemap

class StaticViewSitemap(Sitemap):
    priority = 0.5
    changefreq= 'weekly'
    def items(self):
        return [
            'main:index',
            'test1:index',
            'test2:index',
            'test3:index',
            'test4:index',
        ]
    def location(self, item):
        return reverse(item)