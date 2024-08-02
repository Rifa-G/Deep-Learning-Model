import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, r2_score, mean_absolute_percentage_error, explained_variance_score, confusion_matrix

# Fetch data
start = '2012-01-01'
end = '2022-12-21'
stock = 'VDE'  # Change to Vanguard Energy ETF (VDE)
data = yf.download(stock, start, end)
data.reset_index(inplace=True)

# Plot 100-day moving average
ma_100_days = data.Close.rolling(100).mean()
plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r')
plt.plot(data.Close, 'g')
plt.title('100-day Moving Average vs Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Plot 100-day and 200-day moving average
ma_200_days = data.Close.rolling(200).mean()
plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label='100-day MA')
plt.plot(ma_200_days, 'b', label='200-day MA')
plt.plot(data.Close, 'g', label='Close Price')
plt.title('100-day and 200-day Moving Averages vs Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Drop NaN values
data.dropna(inplace=True)

# Split data into training and testing
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scale = scaler.fit_transform(data_train)

# Create training dataset
x_train = []
y_train = []

for i in range(100, data_train_scale.shape[0]):
    x_train.append(data_train_scale[i-100:i])
    y_train.append(data_train_scale[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Build the model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)

# Model summary
model.summary()

# Prepare test data
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

x_test = []
y_test = []

for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i-100:i])
    y_test.append(data_test_scale[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
y_predict = model.predict(x_test)

# Rescale predictions
scale = 1 / scaler.scale_[0]
y_predict = y_predict * scale
y_test = y_test * scale

# Convert to classification problem (e.g., 1 if price increases, 0 if decreases)
threshold = 0  # Set threshold for classification
y_test_class = (y_test[1:] > y_test[:-1]).astype(int)
y_predict_class = (y_predict[1:] > y_predict[:-1]).astype(int)

# Plot predictions vs actual values
plt.figure(figsize=(10, 8))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_predict, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.title('Stock Price Prediction')
plt.show()

# Save the model
model.save('Stock_Predictions_Model.keras')

# Evaluate the model
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_predict)
mape = mean_absolute_percentage_error(y_test, y_predict)
explained_var = explained_variance_score(y_test, y_predict)
precision = precision_score(y_test_class, y_predict_class)
conf_matrix = confusion_matrix(y_test_class, y_predict_class)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-Squared (R²): {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')
print(f'Explained Variance Score: {explained_var}')
print(f'Precision Score: {precision}')
print(f'Confusion Matrix:\n{conf_matrix}')
