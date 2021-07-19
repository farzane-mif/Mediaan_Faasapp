
import time
import datetime
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
import streamlit as st
from PIL import Image 
from sklearn.metrics import mean_squared_error

os.chdir('C:/Users/FarzanehAkhbar/Documents/FAAS/bitcoin/New folder/bitcoin-predict-master/data')



# ## Data Exploration
data = pd.read_csv("bitcoin.csv")
data = data.sort_values('Date')
data['Date']= pd.to_datetime(data['Date'])

# data['Date'].min()
# data['Date'].max()

st.header("Pridicting Bitcoin Price with LSTM")

original_title = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Select Your Horizon</p>'
st.markdown(original_title, unsafe_allow_html=True)


p1 = st.date_input("Start Date",datetime.date(2014, 12, 1))
p2 = st.date_input("End Date",datetime.date(2020, 1, 1))
p1 = pd.to_datetime(p1)
p2 =  pd.to_datetime(p2)
data = data.loc[(data['Date'] > p1) & (data['Date'] < p2)]

minval = data['Date'].min()
maxval = data['Date'].max()
# st.markdown(f"min date: **{minval}**")
# st.markdown(f"max date: **{maxval}**")
price = data[['Close']]

priceshape = price.shape[0]
# st.markdown(f"priceshape: *{priceshape}")

def draw():  
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.Date, data.Close, color='tab:blue', label='price')
    ax.set_xlabel('Date', )
    ax.set_ylabel('Price', )
    ax.set_title('Bitcoin price',  fontweight='bold')
    ax.grid(True)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    st.pyplot(fig)
    
    
    
if st.button('Plot Data '):
    draw()

# def plot(price):
    
# Normalization
min_max_scaler = MinMaxScaler()
norm_data = min_max_scaler.fit_transform(price.values)

# # Data split
def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

past_history = 5
future_target = 0

#asking user for test/train split threshold
tx = '<p style="font-family:Courier; color:Blue; font-size: 20px;"> </p>'
st.markdown(tx, unsafe_allow_html=True)
tx1 = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Enter test/train split threshold:</p>'
st.markdown(tx1, unsafe_allow_html=True)
threshold = st.number_input("", value=0.8, step=0.1)
# st.markdown(f"threshold: **{threshold}**")

tx3 = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Enter Future Prediction Horizon: </p>'
st.markdown(tx3, unsafe_allow_html=True)
num_prediction = st.number_input("", value=15)


TRAIN_SPLIT = int(len(norm_data) * (threshold))
# st.markdown(f": **{TRAIN_SPLIT}**")


x_train, y_train = univariate_data(norm_data, 0, TRAIN_SPLIT, past_history, future_target)
x_test, y_test = univariate_data(norm_data, TRAIN_SPLIT, None, past_history, future_target)


# Build the model
num_units = 64
learning_rate = 0.0001
activation_function = 'sigmoid'
adam = Adam(lr=learning_rate)
loss_function = 'mse'
batch_size = 5
num_epochs = 50


# Initialize the RNN
model = Sequential()
model.add(LSTM(units = num_units, activation=activation_function, input_shape=(None, 1)))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.1))
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer=adam, loss=loss_function)
  
    # my_bar = st.progress(0)     
def plot():
  # Using the training set to train the model
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=num_epochs,
        shuffle=False
    )  
    original = pd.DataFrame(min_max_scaler.inverse_transform(y_test))
    predictions = pd.DataFrame(min_max_scaler.inverse_transform(model.predict(x_test)))
    
    rmse = (mean_squared_error(original[0], predictions[0]))**(1/2)

    fig, ax = plt.subplots(figsize=(12, 6))
    # ax = sns.lineplot(x=data.Date, y=original[0], label="Actual", color='royalblue')
    # ax = sns.lineplot(x=data.Date, y=predictions[0], label="Prediction", color='tomato')
    
    ax.plot(data['Date'][-len((original)):], original[0], label="Actual", color='royalblue')
    ax.plot(data['Date'][-len((original)):], predictions[0], label="Prediction", color='tomato')
    
    
    ax.set_title('Bitcoin price vs Predicted Price', size = 14, fontweight='bold')
    ax.set_xlabel("Days", size = 14)
    ax.set_ylabel("Price (USD)", size = 14)
    ax.grid(True)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    st.pyplot(fig)
    
    
    
    return (rmse,model)

# num_prediction = 30  



def predictfuture(num_prediction, model):
    tex = '<p style="font-family:Courier;text-align: center; color:Black; font-size: 20px;background-color: lightgreen"; text-align: center; padding-left: 1rem>        Future Bitcoin price prediction</p>'
    st.markdown(tex , unsafe_allow_html=True)
    
    def unseendaysprediction(num_prediction, model):
        prediction_list = norm_data[-past_history:]
        
        for _ in range(num_prediction):
            x = prediction_list[-past_history:]
            x = x.reshape((1, past_history, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[past_history-1:]
            
        return prediction_list
    
    
    
    forecast = unseendaysprediction(num_prediction, model)
    
    arr = forecast.reshape(-1, 1) 
    unseenforcast = pd.DataFrame(min_max_scaler.inverse_transform(arr))
    unseenforcast.columns= ['val']
    
    def unseendays_dates(num_prediction):
        last_date = data['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
        return prediction_dates
    
    forecast_dates = unseendays_dates(num_prediction)
    
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.lineplot(x=data['Date'], y=data['Close'], label=" Actual", color='royalblue')
    ax = sns.lineplot(x=forecast_dates, y=unseenforcast['val'], label="Prediction", color='tomato')
    ax.grid(True)
    ax.set_title('Bitcoin price', size = 14, fontweight='bold')
    ax.set_xlabel("Days", size = 14)
    ax.set_ylabel("Price (USD)", size = 14)
       
    st.pyplot(fig)  


         

if st.button('Plot Real Price vs Predicted'):
    
    tex = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;background-color: yellow"; text-align: center; padding-left: 20px>This will take a while, Please Wait... </p>'
    st.markdown(tex, unsafe_allow_html=True)
    
    rmse , model = plot()
    st.markdown((f"RMSE: {rmse}"))
    predictfuture(num_prediction, model)

    


   








