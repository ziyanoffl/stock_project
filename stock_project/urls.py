from django.urls import path, include # Import the path and include functions

urlpatterns = [
    path('', include('stock_app.urls')), # Add a url pattern that includes your app's urls
]
