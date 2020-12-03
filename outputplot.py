import matplotlib.pyplot as plt
import pandas as pd


dataset_total = pd.read_csv('sggs_load_total_data.csv')



predicted = pd.read_csv('output.csv')


predicted_units = predicted.values



plt.plot(dataset_total["Unnamed: 0"],dataset_total.units,
         color = 'b',label = 'real_data')

plt.plot(predicted["Unnamed: 0"],predicted.predicted_units,
         color = 'r',label = 'predicted_data')

plt.legend()

plt.show()
