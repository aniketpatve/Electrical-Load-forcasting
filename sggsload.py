
def predict_unit(input_date,per_unit_rate = 9.7):
    import pandas as pd
    from tensorflow.keras.models import load_model 
    from datetime import datetime 
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np 
    scale = MinMaxScaler(feature_range = (0,1))
    model = load_model('sggs_load_rnn_model.h5')
    df_predictions = pd.date_range('2020/1/1',periods = 48,freq = 'M')
    fun = [datetime.strftime(x,'%Y-%m-%d') for x in df_predictions]
    dataset_test = pd.read_csv('sggs_test_dataset.csv')
    first_twelve = dataset_test.drop('Unnamed: 0',axis = 1)
    first_twelve = scale.fit_transform(first_twelve)
    op = []
    batch = first_twelve.reshape(1,12,1)
    desired = fun.index(input_date)+1
    for x in range(0,48):
        sk1 = model.predict(batch)
        op.append(scale.inverse_transform(sk1)[0])
        batch = np.append(batch[:,1:,:],sk1.reshape(1,1,1),axis = 1)
        if len(op) == desired :
                break
     
            
    return op[-1],op[-1]*per_unit_rate


    
def predict_kva(input_date,kva_rate = 391) :
    import pandas as pd
    from tensorflow.keras.models import load_model 
    from datetime import datetime 
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np 
    scale = MinMaxScaler(feature_range = (0,1))
    model = load_model('Demand_kva.h5')
    df_predictions = pd.date_range('2020/1/1',periods = 48,freq = 'M')
    fun = [datetime.strftime(x,'%Y-%m-%d') for x in df_predictions]
    dataset_test = pd.read_csv('kva_test.csv')
    first_twelve = dataset_test.drop('Unnamed: 0',axis = 1)
    first_twelve = scale.fit_transform(first_twelve)
    op = []
    batch = first_twelve.reshape(1,12,1)
    desired = fun.index(input_date)+1
    for x in range(0,48):
        sk1 = model.predict(batch)
        op.append(scale.inverse_transform(sk1)[0])
        batch = np.append(batch[:,1:,:],sk1.reshape(1,1,1),axis = 1)
        if len(op) == desired :
                break
    return op[-1],op[-1]*kva_rate
    

def a_zone(input_date,a_zone_rate = -0.5):
    import pandas as pd
    from tensorflow.keras.models import load_model 
    from datetime import datetime 
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np 
    scale = MinMaxScaler(feature_range = (0,1))
    model = load_model('a_zone.h5')
    df_predictions = pd.date_range('2020/1/1',periods = 48,freq = 'M')
    fun = [datetime.strftime(x,'%Y-%m-%d') for x in df_predictions]
    dataset_test = pd.read_csv('a_zone_test.csv')
    first_twelve = dataset_test.drop('Unnamed: 0',axis = 1)
    first_twelve = scale.fit_transform(first_twelve)
    op = []
    batch = first_twelve.reshape(1,12,1)
    desired = fun.index(input_date)+1
    for x in range(0,48):
        sk1 = model.predict(batch)
        op.append(scale.inverse_transform(sk1)[0])
        batch = np.append(batch[:,1:,:],sk1.reshape(1,1,1),axis = 1)
        if len(op) == desired :
                break
    return op[-1],op[-1]*a_zone_rate



def b_zone(input_date,b_zone_rate = 0):
    
    import pandas as pd
    from tensorflow.keras.models import load_model 
    from datetime import datetime 
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np 
    scale = MinMaxScaler(feature_range = (0,1))
    model = load_model('b_zone_rnn_model.h5')
    df_predictions = pd.date_range('2020/1/1',periods = 48,freq = 'M')
    fun = [datetime.strftime(x,'%Y-%m-%d') for x in df_predictions]
    dataset_test = pd.read_csv('b_zone_test.csv')
    first_twelve = dataset_test.drop('Unnamed: 0',axis = 1)
    first_twelve = scale.fit_transform(first_twelve)
    op = []
    batch = first_twelve.reshape(1,12,1)
    desired = fun.index(input_date)+1
    for x in range(0,48):
        sk1 = model.predict(batch)
        op.append(scale.inverse_transform(sk1)[0])
        batch = np.append(batch[:,1:,:],sk1.reshape(1,1,1),axis = 1)
        if len(op) == desired :
                break
    return op[-1],op[-1]*b_zone_rate

def c_zone(input_date,c_zone_rate = 0.8):
    import pandas as pd
    from tensorflow.keras.models import load_model 
    from datetime import datetime 
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np 
    scale = MinMaxScaler(feature_range = (0,1))
    model = load_model('c_zone_rnn_model.h5')
    df_predictions = pd.date_range('2020/1/1',periods = 48,freq = 'M')
    fun = [datetime.strftime(x,'%Y-%m-%d') for x in df_predictions]
    dataset_test = pd.read_csv('c_zone_test.csv')
    first_twelve = dataset_test.drop('Unnamed: 0',axis = 1)
    first_twelve = scale.fit_transform(first_twelve)
    op = []
    batch = first_twelve.reshape(1,12,1)
    desired = fun.index(input_date)+1
    for x in range(0,48):
        sk1 = model.predict(batch)
        op.append(scale.inverse_transform(sk1)[0])
        batch = np.append(batch[:,1:,:],sk1.reshape(1,1,1),axis = 1)
        if len(op) == desired :
                break
    return op[-1],op[-1]*c_zone_rate


def d_zone(input_date,d_zone_rate = 1.10):
    
    import pandas as pd
    from tensorflow.keras.models import load_model 
    from datetime import datetime 
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np 
    scale = MinMaxScaler(feature_range = (0,1))
    model = load_model('d_zone_rnn_model.h5')
    df_predictions = pd.date_range('2020/1/1',periods = 48,freq = 'M')
    fun = [datetime.strftime(x,'%Y-%m-%d') for x in df_predictions]
    dataset_test = pd.read_csv('d_zone_test.csv')
    first_twelve = dataset_test.drop('Unnamed: 0',axis = 1)
    first_twelve = scale.fit_transform(first_twelve)
    op = []
    batch = first_twelve.reshape(1,12,1)
    desired = fun.index(input_date)+1
    for x in range(0,48):
        sk1 = model.predict(batch)
        op.append(scale.inverse_transform(sk1)[0])
        batch = np.append(batch[:,1:,:],sk1.reshape(1,1,1),axis = 1)
        if len(op) == desired :
                break
    return op[-1],op[-1]*d_zone_rate



 
def generate_bill(input_date,per_unit_rate = 9.7,
    kva_rate = 391,a_zone_rate = -1.05,b_zone_rate = 0,c_zone_rate = 0.80 ,
    d_zone_rate = 1.10 ,wheeling_rate = 0.76,
    fac_rate = 1.35,tax_on_sale_rate = 9.04,charges_for_excess_demand = 1.1,
    power_factor = 0.995 ) :
    (units,Energy_charges) = predict_unit(input_date,per_unit_rate)
    (kva,Demand_charges) = predict_kva(input_date,kva_rate)
    (a_zone_units,cost_a_zone) = a_zone(input_date,a_zone_rate)
    (b_zone_units,cost_b_zone) = b_zone(input_date,b_zone_rate)
    (c_zone_units,cost_c_zone) = c_zone(input_date,c_zone_rate)
    (d_zone_units,cost_d_zone) = d_zone(input_date,c_zone_rate)
    
    percentage_tariff = 0.035
    wheeling_charges = wheeling_rate * units 
    Tod_tariff = cost_a_zone+cost_b_zone+cost_c_zone+cost_d_zone
    power_factor_charges_incentives = -(percentage_tariff * Energy_charges)
    tax_on_sale_charges =  tax_on_sale_rate*units 
    
    total_bill = Energy_charges+Demand_charges+cost_a_zone+cost_b_zone+cost_c_zone+cost_d_zone+ wheeling_charges+Tod_tariff+power_factor_charges_incentives+tax_on_sale_charges
    
    trans_1 = 400
    trans_2 = 315 
    trans_3 = 250 
    
    
    Contract_demand = 400
    
    print("BILLING DETAILS :")
    print("-"*50)
    print("Contract Demand : {} ".format(Contract_demand))
    print("-"*50)
    print("predicted_KVA : {}".format(kva))
    print("-"*50)
    print("Demand Charges : {} ".format(Demand_charges))
    print("-"*50)
    print("Wheeling Charges : {} ".format(wheeling_charges))
    print("-"*50)
    print("Energy Charges : {} ".format(Energy_charges))
    print("-"*50)  
    print("TOD TARIFF EC : {}".format(Tod_tariff))
    print("-"*50)
    print("Power Factor Tariff or Incentives:{} ".format(power_factor_charges_incentives))
    print("-"*50)
    print("Tax on sale : {}".format(tax_on_sale_charges))
    print("-"*50)
    print("TOTAL BILL AMOUNT : {} in rs only".format(total_bill))
    print("-"*50)
    print("AVAILABLE TRANSFORMERS IN SUBSTATION : ")
    print("-"*50)
    print("Transformer 1 : {} KVA" .format(trans_1))
    print("-"*50)
    print("Transformer 2 : {} KVA".format(trans_2))
    print("-"*50)
    print("Transformer 3 : {} KVA".format(trans_3))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
      
       
        


       
    

    
    
    
    
    
    
    
    
    






