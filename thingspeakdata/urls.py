from django.urls import path
from .views import *

urlpatterns = [
    path('', dashboard),
    path('temperature/', temperature),
    path('humidity/', humidity),
    path('out_temp/', out_temp),
    path('pressure/', pressure),
]