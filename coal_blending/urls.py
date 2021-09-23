from django.urls import path

from . import views

app_name = 'coal_blending'
urlpatterns = [
    path('', views.redirect_home, name='home'),
    path('data_input/', views.DataInputView.as_view(), name='data_input'),
]
