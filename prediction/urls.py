from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name= 'index'),
    path('roccurve/', views.roccurve, name= 'roccurve'),
    path('compare/', views.compare, name= 'compare'),
    path('chart/', views.pie_chart, name= 'chart'),
    path('matrix/', views.con_matrix, name= 'conmatrix'),

]
