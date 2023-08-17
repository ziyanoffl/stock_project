from django.urls import path  # Import the path function
from .views import home_view, stock_view, find_suitable_stocks, \
    stock_suggestion, what_if, what_if_results  # Import the home_view and stock_view functions

urlpatterns = [
    path('', home_view, name='home_view'),  # Add a url pattern for your home_view function
    path('stock/', stock_view, name='stock_view'),  # Add a url pattern for your stock_view function
    path('find_suitable_stocks/', find_suitable_stocks, name='find_suitable_stocks'),
    path('stock_suggestion/', stock_suggestion, name='stock_suggestion'),
    path('what_if/', what_if, name='what_if'),
    path('what_if_results/', what_if_results, name='what_if_results'),
]
