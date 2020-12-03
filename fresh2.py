

import pandas  as pd 
import numpy as np 
import matplotlib.pyplot as plt 



total_data = pd.read_csv("sggs_load_total_data.csv")

total_data = total_data.drop('Unnamed: 0',axis = 1)

dataset_train = pd.read_csv('sggs_train_dataset.csv') 

dataset_train = dataset_train.drop('Unnamed: 0',axis = 1)

training_set = dataset_train[["units"]].values




from sklearn.preprocessing import MinMaxScaler 

scale = MinMaxScaler(feature_range = (0,1))

training_set_scaled = scale.fit_transform(training_set)






X_train = []
y_train = []

for x in range(12,len(training_set_scaled)):
     X_train.append(training_set_scaled[x-12:x,0])
     y_train.append(training_set_scaled[x,0])
    
X_train , y_train = np.array(X_train) , np.array(y_train)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)


from tensorflow.keras.layers import Dense,Dropout,LSTM 
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import mean_absolute_error,mse 
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split 
from  tensorflow.keras.models import Sequential 

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





dataset_test = pd.read_csv('sggs_test_dataset.csv')

dataset_test = dataset_test.drop('Unnamed: 0',axis = 1)

test = dataset_test[["units"]].values 


test = scale.fit_transform(test)

total_data_scaled = np.append(training_set_scaled,test)


inputs = total_data_scaled[40:52] 





time = 64 
test_pr = inputs.reshape(1,12,1)

p = []


for x in range(0,time):
   sk = reg.predict(test_pr)    
   p.append(scale.inverse_transform(sk)[0])
   test_pr = np.append(test_pr[:,1:,:],sk.reshape(1,1,1),axis = 1)
    

df = pd.DataFrame()

df_date = pd.date_range('2018/1/1',periods = time,freq = 'M')
   
for x in range(0,len(p)):
    some = pd.DataFrame(p[x],columns = ["predicted_units"])
    df = pd.concat([df,some],axis = 0)

df.index = df_date

predicted_units = df.values

total_data = pd.read_csv("sggs_load_total_data.csv")

total_data = total_data.drop('Unnamed: 0',axis = 1)

total_data.index = pd.date_range('2013/9/1',periods = 76,freq = 'M')

real_units = total_data.values 

plt.plot(total_data.index,real_units,color = 'red',label = 'Real units')

plt.plot(df.index,predicted_units,color = 'blue',label = 'Predicted units')

plt.legend()

plt.show()










    




    

    

    

    

    
    
    
















    





