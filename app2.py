import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)

#Describing Data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, 'b', label = 'Closing Price')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig)

#100 MA
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'y', label = '100MA')
plt.plot(df.Close, 'b', label = 'Closing Price')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig)

#200 MA
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'y', label = '100MA')
plt.plot(ma200, 'r', label = '200MA')
plt.plot(df.Close, 'b', label = 'Closing Price')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig)

# Splitting Data into Training and Testing
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) 
#data_train and data_test are variables
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))]) 
print(data_train.shape)
print(data_test.shape)

#Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1)) 
#scaler is defined by an object which is scale down the data between 0 and 1

data_train_array = scaler.fit_transform(data_train) 
#data_train_array is an variable wchih store the scaled down data of data_train

#Load Model
model = load_model('stock_predict_keras_model.h5')

#Testing
past_100_days = data_train.tail(100)
final_df = pd.concat([past_100_days,data_test], ignore_index = True)

#now we have to scale down the data of final_df
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

#Making Prediction
y_predict = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predict = y_predict * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Prediction VS Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predict, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)