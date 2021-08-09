from django.urls import path
from .views import *

urlpatterns = [
    path('', temperature),
    path('temperature/', temperature),
    path('humidity/', humidity),
    path('out_temp/', out_temp),
    path('pressure/', pressure),
]