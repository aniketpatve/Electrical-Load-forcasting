import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.layers import Dense,Dropout,LSTM 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.activations import relu,linear 


scale = MinMaxScaler(feature_range = (0,1))

df = pd.read_csv("kva.csv")
df = df.drop(df[["Unnamed: 0"]],axis = 1)
df = df.fillna(method = 'ffill')

training_set = pd.read_csv("kva_train.csv")
training_set = training_set.drop(training_set[["Unnamed: 0"]],axis = 1)

test_set = pd.read_csv("kva_test.csv")
test_set = test_set.drop(test_set[["Unnamed: 0"]],axis = 1)

training_set_scaled = scale.fit_transform(training_set)
test_set_scaled = scale.fit_transform(test_set)

X_train = [] 
y_train = [] 



total_data_scaled = scale.fit_transform(df)

for x in range(12,len(total_data_scaled)): 
    X_train.append(total_data_scaled[x-12:x])
    y_train.append(total_data_scaled[x])
    
X_train = np.array(X_train)
y_train = np.array(y_train)

kva = Sequential() 

kva.add(LSTM(units = 150,input_shape = (X_train.shape[1],1) ,activation = relu,return_sequences = True 
             ))

kva.add(Dropout(0.5))

kva.add(LSTM(units = 50,activation = relu,return_sequences = True 
             ))

kva.add(Dropout(0.5))

kva.add(LSTM(units = 50,activation = relu,
             return_sequences = False
             ))

kva.add(Dropout(0.5))
kva.add(Dense(units = 1,activation = linear))

kva.compile(optimizer = "adam",loss = "mean_squared_error")

kva.fit(X_train,y_train,epochs = 500,batch_size = 12)

batch = training_set_scaled[-12:].reshape(1,12,1)

time = 48

pred_kva = []


for x in range(0,time):
    kv = kva.predict(batch)
    pred_kva.append(scale.inverse_transform(kv))
    batch = np.append(batch[:,1:,:],kv.reshape(1,1,1))
    batch = batch.reshape(1,12,1)
    




df_date = pd.date_range('2013/9/1',periods = 76,freq = "M")

df_date_pred = pd.date_range('2019/1/1',periods = time,freq = "M")

df_p = pd.DataFrame()


   
for x in range(0,len(pred_kva)):
    lime = pd.DataFrame(pred_kva[x],columns = ["predicted_units"])
    df_p = pd.concat([df_p,lime],axis = 0)


plt.plot(df_date,df,color = 'blue')

plt.plot(df_date_pred,df_p,color = 'black')

plt.title("predicted_kva")

plt.show()

