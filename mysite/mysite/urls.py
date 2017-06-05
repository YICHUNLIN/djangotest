from django.conf.urls import include, url
from django.contrib import admin
from QR.views import qrindex
urlpatterns = [
    # Examples:
    # url(r'^$', 'mysite.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    url(r'^qr/', qrindex),
]
