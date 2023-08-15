import os

from django.shortcuts import render  # Import the HttpResponse and render classes
from django.shortcuts import HttpResponse
import yfinance as yf  # Import the yfinance module
import numpy as np
import pandas as pd  # Import the pandas module
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression  # Import the LinearRegression class
from sklearn.model_selection import train_test_split
from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter
import datetime as dt
from stock_project import settings


def home_view(request):  # Define a home_view function that takes a request as an argument
    # Get the stock symbol from the request and assign it to a variable. If it is None, set it to some default value.
    stock = request.GET.get('stock')
    if not stock:
        stock = 'AAPL'  # Default stock symbol
    return render(request, 'home.html',
                  {'stock': stock})  # Render the home.html template and pass the stock variable as context


def stock_view(request):  # Define a stock_view function that takes a request and a stock symbol as arguments
    stock = request.GET.get('stock')
    # Get the investment amount and days from the request and convert them to float and int respectively. If they are None, set them to some default values.
    investment = request.GET.get('investment')
    days = request.GET.get('days')
    if investment:
        investment = float(investment)
    else:
        investment = 1000  # Default investment amount
    if days:
        days = int(days)
    else:
        days = 10  # Default number of days

    # Download the stock data as a pandas dataframe using the yfinance.download function. You don't need to save it to the database.
    from datetime import datetime, timedelta

    # Get yesterday's date
    # yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # Replace 'end' parameter with yesterday's date
    df = yf.download(stock, period='2y', interval='1d')
    print(df)

    # Fetching ticker values from Yahoo Finance API
    df_ml = df[['Adj Close']]
    forecast_out = int(days)
    df_ml['Prediction'] = df_ml[['Adj Close']].shift(-forecast_out)
    # Splitting data for Test and Train
    X = np.array(df_ml.drop(['Prediction'], axis=1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df_ml['Prediction'])
    y = y[:-forecast_out]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # Applying Linear Regression
    clf = LinearRegression()
    clf.fit(X_train, y_train)

    # Prediction Score
    confidence = clf.score(X_test, y_test)
    # Predicting for 'n' days stock data
    forecast_prediction = clf.predict(X_forecast)
    forecast = forecast_prediction.tolist()

    # Evaluate the model performance
    pred_dict = {"Date": [], "Prediction": []}
    for i in range(0, len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i])

    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="lightgrey", plot_bgcolor="lightgrey", font_color="black")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

    # Get the share price when invested from yesterday's closing price which should be today's opening price
    price_invested = df.iloc[-1]['Adj Close']

    # Assuming X is a numpy array that stores the original data
    X = np.array(df_ml['Adj Close'])

    # Assuming forecast is a list that stores the scaled predictions
    forecast = forecast_prediction.tolist()

    # Get the last predicted value
    last_prediction = forecast[-1]

    # Get the standard deviation and mean of the original scaled data (X_train)
    std_train = np.std(X_train)
    mean_train = np.mean(X_train)

    # Rescale the last predicted value using the same scaling factors
    price_sold = (last_prediction * std_train) + mean_train

    # Calculate the profit or loss based on the new method
    shares = investment / price_invested  # Number of shares bought with the investment
    profit_loss = (price_sold - price_invested) * shares  # Profit or loss amount
    profit_loss_percent = (profit_loss / investment) * 100  # Profit or loss percentage
    final_value = price_sold * shares

    # Create a list of results to pass to the template.
    results = [round(price_invested, 2), round(price_sold, 2), round(final_value, 2), round(profit_loss, 2),
               round(profit_loss_percent, 2)]

    # Create a context dictionary to pass to the template.
    context = {
        'stock': stock,
        'investment': investment,
        'days': days,
        'results': results,
        'profit_loss_percent': round(profit_loss_percent, 2),
        'profit_loss': round(profit_loss, 2),
        # 'MSE': round(mse, 2),
        # 'R2': round(r2, 2),
        'plot_div_pred': plot_div_pred,
        # 'predicted_chart': 'predicted_prices.png',
    }

    # Render the template with the context using the render function.
    return render(request, 'stock.html', context)
