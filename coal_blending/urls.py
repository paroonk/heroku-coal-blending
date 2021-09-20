from django.urls import path

from . import views

app_name = 'coal_blending'
urlpatterns = [
    path('', views.redirect_home, name='home'),
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
]
