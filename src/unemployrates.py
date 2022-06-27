import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from sklearn.metrics import mean_squared_error
import sklearn

numOfNewDataPoints = input("number of new data points. More you do less accurate it gets")
numOfNewDataPoints = int(numOfNewDataPoints)

df = pd.read_csv('../data/USunemploytraining.csv')
data=df.filter(['unemployment rate'])
print(data)

plt.figure(figsize=(16,8))
plt.title('unemployment rate history')
plt.plot(data['unemployment rate'])
plt.xlabel('DATE', fontsize=18)
plt.ylabel('unemployment rate history (%)', fontsize=18)
plt.show()

dataset = data.values
training_data_len = math.ceil(len(dataset) * 0.8)

print(training_data_len)

# scale data

scaler = MinMaxScaler(feature_range=(0,1))
#scaled_data = pd.DataFrame(scaler.fit_transform(data),columns = data.columns)
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)

# create training data set 

train_data = scaled_data[0: training_data_len , :]
# spilt data into x_train and y_train

x_train = []
y_train = []

for i in range(100, len(train_data)):
    x_train.append(train_data[i-100:i, 0])
    y_train.append(train_data[i, 0])
    if i<=101:
        print(x_train)
        print(y_train)

#convert x_train and y_train to numpy array

x_train, y_train = np.array(x_train), np.array(y_train)

# reshape the data
x_train = np.reshape(x_train, (605, 100, 1))
x_train.shape

# build LSTM model

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50, return_sequences=False))
#model.add(Dense(512))
#model.add(Dense(256))
model.add(Dense(25))
model.add(Dense(1))

print("hello")

model.compile(optimizer='adam', loss='mean_squared_error')

# train

model.fit(x_train, y_train, batch_size=1, epochs=1)

# create the testing data set

test_data = scaled_data[training_data_len - 100: , :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(100, len(test_data)):
  x_test.append(test_data[i-100:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test = y_test.reshape(y_test.size, 1)

rmse=np.sqrt( np.mean((predictions - y_test)**2))
rmse

train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions

#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD($)',fontsize=18)
plt.plot(train['unemployment rate'])
plt.plot(valid[['unemployment rate','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()

# predict future days 

prediction_days = 100
model_inputs = dataset[len(dataset) - len(test_data) - 100:]
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

predictions1 = model.predict(real_data)
predictions1 = scaler.inverse_transform(predictions1)
print(f"predictions1: {predictions1}")  
predictions1 = predictions1[0][0]
print(predictions1)
predictions1 = predictions1
df = pd.read_csv('/content/drive/MyDrive/USunemploytraining copy.csv')
data=df.filter(['unemployment rate'])
xs = df.filter(['unemployment rate'])
print(data)
print(df)
predictionsa = [881, predictions1]
df.append(predictionsa)
print(df)

predictionsb = [predictions1]
a_series = pd.Series(predictionsb, index = data.columns)
data = data.append(a_series, ignore_index=True)
print(data)

prediction_days = 100
dataset = data.values

model_inputs = dataset[len(dataset) - len(test_data) - 100:]
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

predictions2 = model.predict(real_data)
predictions2 = scaler.inverse_transform(predictions2)
print(f"predictions2: {predictions2}")  
predictions2 = predictions2[0][0]
print(predictions2)
predictionsc = predictions2
a_series = pd.Series(predictionsc, index = data.columns)
data = data.append(a_series, ignore_index=True)

print(data)

for i in range(numOfNewDataPoints):
  prediction_days = 100
  dataset = data.values

  model_inputs = dataset[len(dataset) - len(test_data) - 100:]
  model_inputs = model_inputs.reshape(-1, 1)
  model_inputs = scaler.transform(model_inputs)
  real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
  real_data = np.array(real_data)
  real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

  predictions10 = model.predict(real_data)
  predictions10 = scaler.inverse_transform(predictions10)
  print(f"predictions10: {predictions10}")  
  predictions10 = predictions10[0][0]
  print(predictions10)
  predictionsc = predictions10
  a_series = pd.Series(predictionsc, index = data.columns)
  data = data.append(a_series, ignore_index=True)

  print(data)

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date (blue is predicited)',fontsize=18)
plt.ylabel('Unemployment rates',fontsize=18)
plt.plot(data)
plt.plot(xs)
# final output
plt.show()
