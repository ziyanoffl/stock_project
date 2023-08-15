from django.urls import path # Import the path function
from .views import home_view, stock_view # Import the home_view and stock_view functions

urlpatterns = [
    path('', home_view, name='home_view'), # Add a url pattern for your home_view function
    path('stock/?', stock_view, name='stock_view'), # Add a url pattern for your stock_view function
]
