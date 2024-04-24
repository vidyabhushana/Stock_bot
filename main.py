import numpy as np
from pandas_datareader import data as pdr
from datetime import datetime
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
from flask_socketio import SocketIO, emit 
import threading
import time
from flask import Flask, render_template, jsonify, request
from yaml import emit
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objs as go 
import plotly.io as pio
from flask_cors import CORS
app = Flask(__name__)
socketio = SocketIO(app, mode='threading')
CORS(app)

headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiYjk0NzQyNTctOWY4Yy00Y2I0LWE5Y2UtMjNhM2Q5ZDM4ZTRhIiwidHlwZSI6ImFwaV90b2tlbiJ9.Hc9nYSzzt5thtOWuWq5Kr14wD0nixZdO7ODBheS3lLE"
}

url = "https://api.edenai.run/v2/text/question_answer"

def fetch_historical_data(symbol, period="7d", interval="1m"):
    data = yf.download(tickers=symbol, period=period, interval=interval)
    return data

def calculate_indicator(data, indicator='', length=14):
    if indicator == 'SMA':
        data[indicator] = ta.sma(data['Close'], length=length)
    if indicator == 'RSI':
        data[indicator] = ta.rsi(data['Close'], length=length)
    if indicator == 'EMA':
        data[indicator] = ta.ema(data['Close'], length=length)

def plot_indicator(data, indicator, model='linear', filename='plot.png', symbol=''):
    history = data[indicator].tail(100).reset_index()
    history['index'] = history.index

    if model == 'linear':
        reg = LinearRegression()
    elif model == 'svr':
        reg = SVR(kernel='rbf', C=1e3, gamma=0.1)
        scaler = StandardScaler()
        history[indicator] = scaler.fit_transform(history[[indicator]])

    reg.fit(history[['index']], history[indicator])
    history[f'{model}_pred'] = reg.predict(history[['index']])

    sns.set(style="whitegrid")
    plt.figure(figsize=(20, 16))
    sns.lineplot(x='index', y=indicator, data=history)
    sns.lineplot(x='index', y=f'{model}_pred', data=history, color='red', label=f'{model} regression')
    plt.title(f"{indicator} with {model.capitalize()} Regression for {symbol}")
    plt.xlabel("Date and Time")
    plt.ylabel(indicator)
    plt.xticks(np.arange(0, len(history), 5), history['Datetime'].iloc[::5].dt.strftime('%a.- (%d- %b- %Y) %H:%M'), rotation=45)
    plt.legend()

    plt.savefig(filename)
    plt.close()

    return filename

from flask import after_this_request

@app.route('/technical', methods=['GET'])
def plot_technical():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400

    data = fetch_historical_data(symbol)
    if data.empty:
        return jsonify({"error": "No data available for the given symbol"}), 404
    
    calculate_indicator(data, 'RSI')
    calculate_indicator(data, 'SMA')
    calculate_indicator(data, 'EMA')
    plot_files = [
        plot_indicator(data, 'SMA', filename=f'static/{symbol}_SMA_plot.png', symbol=symbol),
        plot_indicator(data, 'RSI', filename=f'static/{symbol}_RSI_plot.png', symbol=symbol),
        plot_indicator(data, 'EMA', filename=f'static/{symbol}_EMA_plot.png', symbol=symbol)
    ]
    # Return the plot file paths as JSON
    return jsonify(SMA_plot=plot_files[0], RSI_plot=plot_files[1],EMA_plot=plot_files[2])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about', methods=['GET'])
def get_about_info():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400

    try:
        # Fetch stock information using yfinance
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Extract required fields from the info dictionary
        about_info = {
            "Name": info.get('longName', ''),
            "Symbol": info.get('symbol', ''),
            "Description": info.get('longBusinessSummary', ''),
            "AssetType": info.get('quoteType', ''),
            "Exchange": info.get('exchange', ''),
            "Industry": info.get('industry', ''),
            "Country": info.get('country', ''),
            "Currency": info.get('currency', ''),
            "MarketCapitalization": info.get('marketCap', ''),
            "SharesOutstanding": info.get('sharesOutstanding', ''),
            "CIK": info.get('CIK', ''),
            "Sector": info.get('sector', ''),
            "Address": info.get('address1', '') + ', ' + info.get('city', '') + ', ' + info.get('state', '') + ', ' + info.get('zip', ''),
            "FiscalYearEnd": info.get('lastFiscalYearEnd', ''),
            "LatestQuarter": info.get('latestQuarter', ''),
            "EBITDA": info.get('EBITDA', ''),
            "PERatio": info.get('trailingPE', ''),
            "PEGRatio": info.get('pegRatio', ''),
            "BookValue": info.get('bookValue', ''),
            "DividendPerShare": info.get('dividendRate', ''),
            "DividendYield": info.get('dividendYield', ''),
            "EPS": info.get('trailingEps', ''),
            "RevenuePerShareTTM": info.get('revenuePerShare', ''),
            "ProfitMargin": info.get('profitMargins', ''),
            "OperatingMarginTTM": info.get('operatingMargins', ''),
            "ReturnOnAssetsTTM": info.get('returnOnAssets', ''),
            "ReturnOnEquityTTM": info.get('returnOnEquity', ''),
            "RevenueTTM": info.get('totalRevenue', ''),
            "GrossProfitTTM": info.get('grossProfit', ''),
            "DilutedEPSTTM": info.get('dilutedEPS', ''),
            "QuarterlyEarningsGrowthYOY": info.get('earningsQuarterlyGrowth', ''),
            "QuarterlyRevenueGrowthYOY": info.get('revenueGrowth', ''),
            "AnalystTargetPrice": info.get('targetMeanPrice', ''),
            "AnalystRatingStrongBuy": info.get('recommendationKey', '') == 'buy',
            "AnalystRatingBuy": info.get('recommendationKey', '') == 'overweight',
            "AnalystRatingHold": info.get('recommendationKey', '') == 'hold',
            "AnalystRatingSell": info.get('recommendationKey', '') == 'underweight',
            "AnalystRatingStrongSell": info.get('recommendationKey', '') == 'sell',
            "TrailingPE": info.get('trailingPE', ''),
            "ForwardPE": info.get('forwardPE', ''),
            "PriceToSalesRatioTTM": info.get('priceToSalesTrailing12Months', ''),
            "PriceToBookRatio": info.get('priceToBook', ''),
            "EVToRevenue": info.get('enterpriseToRevenue', ''),
            "EVToEBITDA": info.get('enterpriseToEbitda', ''),
            "Beta": info.get('beta', ''),
            "52WeekHigh": info.get('fiftyTwoWeekHigh', ''),
            "52WeekLow": info.get('fiftyTwoWeekLow', ''),
            "50DayMovingAverage": info.get('fiftyDayAverage', ''),
            "200DayMovingAverage": info.get('twoHundredDayAverage', ''),
            "DividendDate": info.get('dividendDate', ''),
            "ExDividendDate": info.get('exDividendDate', '')
        }

        return jsonify(about_info)
    except Exception as e:
        print(f"Error fetching about info for {symbol}: {e}")
        return jsonify({"error": f"Failed to fetch about info for {symbol}"}), 500
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def emit_live_data(symbol):
    while True:
        df = yf.download(tickers=symbol, period='1d', interval='1m')
        
        if not df.empty:
            fig = create_candlestick_chart(df, symbol)
            div = pio.to_json(fig)
            
            emit('update_live_data', {'div': div})
        
        time.sleep(30)  # Sleep for 30 seconds before sending the next update

@app.route('/live_data_stream', methods=['GET'])
def live_data_stream():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Stock symbol is missing."}), 400

    threading.Thread(target=emit_live_data, args=(symbol,)).start()
    
    return jsonify({"message": "Live data stream started."})
@app.route('/live_data', methods=['GET'])
def live_data():
    stock = request.args.get('symbol')
    counter = 0 

    if not stock:
        return jsonify({"error": "Stock symbol is missing."}), 400
    
    while counter < 10:  # Limit the loop to 10 seconds
        df = yf.download(tickers=stock, period='1d', interval='1m')
        
        if df.empty:
            return jsonify({"error": "No data available for the given symbol."}), 404

        fig = create_candlestick_chart(df, stock)
        div = pio.to_json(fig)

        time.sleep(1)  # Sleep for 30 seconds before fetching the next data
        counter += 10  # Increment the counter by 30 seconds

    return jsonify({"div": div})

def create_candlestick_chart(df, stock):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'], name='market data'))

    fig.update_layout(
        title=f"{stock} Live Share Price:",
        yaxis_title='Stock Price (USD per Shares)')

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=15, label="15m", step="minute", stepmode="backward"),
                dict(count=45, label="45m", step="minute", stepmode="backward"),
                dict(count=1, label="HTD", step="hour", stepmode="todate"),
                dict(count=3, label="3h", step="hour", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    return fig


@app.route('/run_code', methods=['POST', 'GET'])
def run_code():
    symbol = request.args.get('symbol')
    yf.pdr_override()
    start = pd.Timestamp.now() - pd.DateOffset(years=8)
    end = pd.Timestamp.now()
    df = yf.download(symbol, start=start, end=end)

    data = df.filter(['Close'])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    training_data_len = int(np.ceil(len(dataset) * .70))
    train_data = scaled_data[0:int(training_data_len), :]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions).flatten()

    valid = data[training_data_len:]
    
    # Ensure predictions align with valid data
    predictions = predictions[-len(valid):]

    volume_path, daily_return_path, adj_close_path, prediction_path = generate_plots(df, symbol, valid, predictions)

    valid_json = valid.to_json(orient='index')
    
    # Prepare the data for the table
    actual_data = valid['Close'].values.tolist()
    predicted_data = predictions.tolist()
    date_index = valid.index.strftime('%Y-%m-%d').tolist()

    return jsonify({
        "status": "success",
        "volume_plot": volume_path,
        "daily_return_histogram": daily_return_path,
        "adjusted_close_price": adj_close_path,
        "predicted_vs_actual_plot": prediction_path,
        "data": json.loads(valid_json),
        "table_data": {
            "date": date_index,
            "actual": actual_data,
            "predicted": predicted_data
        }
    })

def generate_plots(df, symbol, valid, predictions):
    # Plot volume
    plt.figure(figsize=(10, 6))
    df['Volume'].plot()
    plt.ylabel('Volume')
    plt.title('Volume')
    volume_path = f'static/{symbol}_volume_plot.png'
    plt.savefig(volume_path)
    plt.close()

    # Plot daily return histogram
    plt.figure(figsize=(10, 6))
    df['Daily Return'] = df['Adj Close'].pct_change()
    df['Daily Return'].hist(bins=50, color='green', alpha=0.7)
    plt.xlabel('Daily Return')
    plt.title('Daily Return Histogram')
    daily_return_path = f'static/{symbol}_daily_return_histogram.png'
    plt.savefig(daily_return_path)
    plt.close()

    # Plot adjusted close price with moving averages
    plt.figure(figsize=(10, 6))
    
    # Plot actual adjusted close price
    plt.plot(df.index, df['Adj Close'], label='Actual Close Price', color='black')
    
    # Calculate and plot moving averages
    df['MA10'] = df['Adj Close'].rolling(window=10).mean()
    df['MA20'] = df['Adj Close'].rolling(window=20).mean()
    df['MA50'] = df['Adj Close'].rolling(window=50).mean()
    
    plt.plot(df.index, df['MA10'], label='MA10', color='blue')
    plt.plot(df.index, df['MA20'], label='MA20', color='green')
    plt.plot(df.index, df['MA50'], label='MA50', color='orange')
    
    plt.ylabel('Adj Close')
    plt.title('Adjusted Close Price with Moving Averages')
    plt.legend()
    adj_close_path = f'static/{symbol}_adjusted_close_price.png'
    plt.savefig(adj_close_path)
    plt.close()

    # Plot for Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.plot(valid.index, valid['Close'], label='Actual Close Price', color='green')
    plt.plot(valid.index, predictions, label='Predicted Close Price', color='red')
    plt.title('Predicted vs Actual Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    prediction_path = f'static/{symbol}_predicted_vs_actual.png'
    plt.savefig(prediction_path)
    plt.close()

    return volume_path, daily_return_path, adj_close_path, prediction_path


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')

    text1 = "i need information related to stocks"
    text2 = "please give me related to stocks only"

    payload = {
        "providers": "openai",
        "texts": [text1, text2],
        "question": question,
        "examples_context": "In 2017, U.S. life expectancy was 78.6 years.",
        "examples": [["What is human life expectancy in the United States?", "78 years."]],
        "fallback_providers": ""
    }

    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        answer = result['openai']['answers'][0]
    else:
        answer = "Sorry, something went wrong."

    return jsonify({'answer': answer})
if __name__ == '__main__':
    socketio.run(app, debug=True)
