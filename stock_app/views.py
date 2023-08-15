import os

from django.shortcuts import render # Import the HttpResponse and render classes
from django.shortcuts import HttpResponse
import yfinance as yf # Import the yfinance module
import numpy as np
import pandas as pd # Import the pandas module
from sklearn.linear_model import LinearRegression # Import the LinearRegression class
from sklearn.model_selection import train_test_split

from stock_project import settings


def home_view(request): # Define a home_view function that takes a request as an argument
    # Get the stock symbol from the request and assign it to a variable. If it is None, set it to some default value.
    stock = request.GET.get('stock')
    if not stock:
        stock = 'AAPL' # Default stock symbol
    return render(request, 'home.html', {'stock': stock}) # Render the home.html template and pass the stock variable as context


def stock_view(request): # Define a stock_view function that takes a request and a stock symbol as arguments
    stock = request.GET.get('stock')
    # Get the investment amount and days from the request and convert them to float and int respectively. If they are None, set them to some default values.
    investment = request.GET.get('investment')
    days = request.GET.get('days')
    if investment:
        investment = float(investment)
    else:
        investment = 1000 # Default investment amount
    if days:
        days = int(days)
    else:
        days = 10 # Default number of days
    
    # Download the stock data as a pandas dataframe using the yfinance.download function. You don't need to save it to the database.
    from datetime import datetime, timedelta

    # Get yesterday's date
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Replace 'end' parameter with yesterday's date
    df = yf.download(stock, start='2021-07-13', end=yesterday, interval='1d')
    print(df)

    # Create a new column for the target variable
    df['Next_Close'] = df['Adj Close'].shift(-1) # The next day's closing price

    # Drop the last row as it has a missing value
    df.interpolate(method='spline', order=3, inplace=True)

    # df.drop('Adj Close', axis=1, inplace=True)

    # Define the features and the target
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']] # You can add more features if you want
    y = df['Next_Close']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model performance
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Create predictor variables for the future dates using linear interpolation
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days, freq='D')  # Generate future dates
    future_X = pd.DataFrame(index=future_dates, columns=X.columns)  # Create an empty dataframe with the same columns as X
    future_X['Open'] = np.interp(future_dates, df.index, df['Open'])  # Interpolate the open prices using the existing data
    future_X['High'] = np.interp(future_dates, df.index, df['High'])  # Interpolate the high prices using the existing data
    future_X['Low'] = np.interp(future_dates, df.index, df['Low'])  # Interpolate the low prices using the existing data
    future_X['Close'] = np.interp(future_dates, df.index, df['Close'])  # Interpolate the close prices using the existing data
    future_X['Volume'] = np.interp(future_dates, df.index, df['Volume'])  # Interpolate the volume using the existing data

    # Predict the closing price for the next n days using the model
    n = days + 1  # Add one more day for the initial investment
    X_new = pd.concat([X.tail(1), future_X])  # Concatenate the last row of X and future_X
    y_new = model.predict(X_new)  # Predict the next n closing prices

    # Get the share price when invested from yesterday's closing price which should be today's opening price
    price_invested = df.iloc[-1]['Adj Close']

    # Get the share price when sold from the last predicted closing price
    price_sold = y_new[-1]

    # Calculate the profit or loss based on the new method
    investment = 1000  # Assume an investment of $1000
    shares = investment / price_invested  # Number of shares bought with the investment
    profit_loss = (price_sold - price_invested) * shares  # Profit or loss amount
    profit_loss_percent = (profit_loss / investment) * 100  # Profit or loss percentage
    final_value = price_sold * shares
    
    # Create a list of results to pass to the template.
    results = [round(price_invested, 2), round(y_new[-1], 2), round(final_value, 2), round(profit_loss, 2), round(profit_loss_percent, 2)]


    import matplotlib.pyplot as plt

    # Create x-axis values (dates)
    dates = X_new.index  # Get the dates from X_new

    # Create a line chart to compare actual and predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(dates[:-days], df['Adj Close'].tail(n - days), color='blue', label='Actual')  # Plot the actual closing prices for the past n-days days
    plt.plot(dates[-days - 1:], y_new[-days - 1:], color='red', label='Predicted')  # Plot the predicted closing prices for the next n days
    plt.title(f'Actual vs Predicted Closing Prices for {stock}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)

    # Define the path to save the chart image in the static folder of the "stock_app" app
    chart_filename = os.path.join(settings.BASE_DIR, 'stock_app', 'static', 'predicted_prices.png')

    # Ensure the directory for the image exists before saving
    os.makedirs(os.path.dirname(chart_filename), exist_ok=True)

    # Save the plot to a file or display it
    plt.savefig(chart_filename)  # Save to a file
    # plt.show()  # Display the plot

    # Create a context dictionary to pass to the template.
    context = {
        'stock': stock,
        'investment': investment,
        'days': days,
        'results': results,
        'profit_loss_percent': round(profit_loss_percent, 2),
        'profit_loss': round(profit_loss, 2),
        'MSE': round(mse, 2),
        'R2': round(r2, 2),
        'predicted_chart': 'predicted_prices.png',
    }

    # Render the template with the context using the render function.
    return render(request, 'stock.html', context)
