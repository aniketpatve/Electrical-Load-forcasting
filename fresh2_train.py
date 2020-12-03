

import pandas  as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,LSTM 
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import mean_absolute_error,mse 
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split 
from  tensorflow.keras.models import Sequential 


total_data = pd.read_csv("sggs_load_total_data.csv")

total_data = total_data.drop('Unnamed: 0',axis = 1)

scale = MinMaxScaler(feature_range = (0,1))

training_set_scaled = scale.fit_transform(total_data)

X_train = []
y_train = []

for x in range(12,len(training_set_scaled)):
     X_train.append(training_set_scaled[x-12:x,0])
     y_train.append(training_set_scaled[x,0])
    
X_train , y_train = np.array(X_train) , np.array(y_train)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)


reg = Sequential([LSTM(units = 100,input_shape = (X_train.shape[1],1),activation = relu,
                       return_sequences = True)])

reg.add(Dropout(0.2))

reg.add(LSTM(units = 50,activation = relu,return_sequences = True))

reg.add(Dropout(0.2))

reg.add(LSTM(units = 50,activation = relu))

reg.add(Dropout(0.2))

reg.add(Dense(units = 1,activation = linear))


reg.compile(optimizer = "adam",loss = "mean_squared_error")

reg.fit(X_train,y_train,epochs = 500,batch_size = 12)

reg.save('sggs_load_rnn_model.h5')





