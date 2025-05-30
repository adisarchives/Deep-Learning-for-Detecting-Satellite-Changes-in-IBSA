from django.urls import path

from . import views

urlpatterns = [path('', views.index, name='home'),  # This will serve the homepage at "/"
               path("index.html", views.index, name="index"),
               path("UserLogin.html", views.UserLogin, name="UserLogin"),	      
               path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
	           path("LoadDataset", views.LoadDataset, name="LoadDataset"),	      
               
               path("TrainCNN", views.TrainCNN, name="TrainCNN"),
               path("DetectSatillite.html", views.PredictImage, name="PredictImage"),
               path("PredictImageAction", views.PredictImageAction, name="PredictImageAction"),           
	       
]