"""
URL configuration for foodspv_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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

from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from django.views.generic import RedirectView
from django.shortcuts import redirect
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('main_app.urls')),
    path('fish/', include('fish_app.urls')),
    path('shrimp/', include('shrimp_app.urls')),
    path('shellfish/', include('shellfish_app.urls')),
    path('login/', auth_views.LoginView.as_view(template_name='main_app/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='main_app/logged_out.html', next_page='/'), name='logout'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

#
handler404 = 'main_app.views.handler404'
handler500 = 'main_app.views.handler500'
