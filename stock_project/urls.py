from django.urls import path, include  # Import the path and include functions

urlpatterns = [
    path('', include('stock_app.urls')),  # Add a url pattern that includes your app's urls
]
handler500 = 'stock_app.views.server_error'
handler404 = 'stock_app.views.page_not_found'
