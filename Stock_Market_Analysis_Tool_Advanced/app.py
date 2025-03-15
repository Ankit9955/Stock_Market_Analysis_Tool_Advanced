import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Fetch real-time stock data (Yahoo Finance API example)
def get_stock_data(symbol, interval='1d', range='30d'):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range}"
    response = requests.get(url).json()
    timestamps = response['chart']['result'][0]['timestamp']
    prices = response['chart']['result'][0]['indicators']['quote'][0]['close']
    return pd.DataFrame({'Date': pd.to_datetime(timestamps, unit='s'), 'Price': prices})

# Get stock data for Google (GOOGL)
df = get_stock_data('GOOGL')

# Data visualization - Stock Price Trend
plt.figure(figsize=(10, 5))
sns.lineplot(x=df['Date'], y=df['Price'], marker='o', color='blue')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.title('Google (GOOGL) Stock Price Trend')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Machine Learning - Simple Stock Price Prediction
df['Days'] = np.arange(len(df)).reshape(-1, 1)  # Convert dates to numeric format
X = df[['Days']]
y = df['Price']
model = LinearRegression()
model.fit(X, y)
future_days = np.array([[len(df) + i] for i in range(1, 6)])  # Predict next 5 days
predictions = model.predict(future_days)

# Plot Predictions
plt.figure(figsize=(8, 4))
plt.plot(df['Days'], df['Price'], label="Historical Prices", color='blue')
plt.plot(future_days, predictions, label="Predicted Prices", linestyle='dashed', color='red')
plt.xlabel('Days')
plt.ylabel('Stock Price ($)')
plt.title('Stock Price Prediction (Next 5 Days)')
plt.legend()
plt.grid()
plt.show()

# Save processed data
df.to_csv('stock_market_data.csv', index=False)

print("Advanced Stock Market Analysis Completed Successfully!")
