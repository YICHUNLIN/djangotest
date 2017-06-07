from django.conf.urls import include, url
from django.contrib import admin
from QR.views import qrindex
from QR.views import doQR
from solveEQU.views import RKindex
from Books.views import bookindex
from Books.views import search

urlpatterns = [
    # Examples:
    # url(r'^$', 'mysite.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    url(r'^qr/', qrindex),
    url(r'^doQR/$', doQR),
    url(r'^rk/', RKindex),
    url(r'^books/', bookindex),
    url(r'^search/$', search)
]
