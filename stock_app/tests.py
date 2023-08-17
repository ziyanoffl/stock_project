from django.test import TestCase, Client
from django.urls import reverse

from stock_app.views import calculate_potential_profit


class TestViews(TestCase):

    def test_home_view(self):
        response = self.client.get(reverse('home_view'))
        self.assertEqual(response.status_code, 200)

    def test_stock_suggestion(self):
        response = self.client.get(reverse('stock_suggestion'))
        self.assertEqual(response.status_code, 200)

    def test_what_if(self):
        response = self.client.get(reverse('what_if'))
        self.assertEqual(response.status_code, 200)

    def test_calculate_potential_profit(self):
        # Define test data
        test_stock = 'AAPL'
        test_target_profit = 100.0
        test_time_interval = 5
        test_investment = 2000.0

        # Call the method under test
        result = calculate_potential_profit(
            stock=test_stock,
            target_profit=test_target_profit,
            time_interval=test_time_interval,
            investment=test_investment
        )

        # Assert the result based on your expected behavior
        self.assertIsInstance(result, (int, float))
        self.assertGreaterEqual(result, 0)  # The result should be non-negative

        # You can also add more specific assertions based on the logic in your method
        # For example, if the potential profit is expected to be greater than or equal to the target profit:
        if result >= test_target_profit:
            self.assertGreaterEqual(result, test_target_profit)
        else:
            self.assertEqual(result, 0)


class FindSuitableStocksTest(TestCase):

    # Set up the test data and objects before running each test
    def setUp(self):
        # Create an instance of Client
        self.client = Client()

        # Define some sample query parameters for testing
        self.query_params = {
            'profit_margin': 50,
            'time_interval': 20,
            'investment': 2000
        }

    # Write a test method to check if the view function returns the correct response
    def test_find_suitable_stocks(self):
        # Create a GET request with the query parameters using the client object
        response = self.client.get('/find_suitable_stocks/', self.query_params)

        self.assertEqual(response.status_code, 200)
