from django.test import TestCase, Client, RequestFactory
from django.urls import reverse

from stock_app.views import calculate_potential_profit, what_if_results


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


class TestWhatIfResultsView(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.factory = RequestFactory()
        self.stock = 'AAPL'
        self.investment = 1000
        self.days = 10
        self.profit_margin = 20
        self.interest_rate_change = 0.01
        self.inflation_change = 0.02
        self.growth_change = 0.03
        self.asset_allocation_change = 0.04
        self.risk_tolerance_change = 0.05

    def test_what_if_results(self):
        request = self.factory.get('/what_if_results/', {
            'stock': self.stock,
            'investment': self.investment,
            'days': self.days,
            'profit_margin': self.profit_margin,
            'interest_rate_change': self.interest_rate_change,
            'inflation_change': self.inflation_change,
            'growth_change': self.growth_change,
            'asset_allocation_change': self.asset_allocation_change,
            'risk_tolerance_change': self.risk_tolerance_change,
        })
        response = what_if_results(request)
        # Check that the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Check that the context contains the expected values
        context = response.context_data
        self.assertEqual(context['stock'], self.stock)
        self.assertEqual(context['days'], int(self.days))
        self.assertEqual(context['profit_margin'], float(self.profit_margin))
        self.assertEqual(context['interest_rate_change'], float(self.interest_rate_change))
        self.assertEqual(context['inflation_change'], float(self.inflation_change))
        self.assertEqual(context['growth_change'], float(self.growth_change))
        self.assertEqual(context['asset_allocation_change'], float(self.asset_allocation_change))
        self.assertEqual(context['risk_tolerance_change'], float(self.risk_tolerance_change))

        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Check if the response content contains specific text or elements
        self.assertContains(response, 'Adjusted Close Price of the Stock')
        self.assertContains(response, 'The predicted closing price')
