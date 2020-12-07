#Bach Project

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import catboost
from tqdm import tqdm 

#%%

df = pd.read_csv("C:/Users/alex/Documents/myPython/RUG/Machine Learning 2020/F.txt", sep="\t", header=None)
df.columns = ['Singer1','Singer2','Singer3','Singer4']
df0 = df.copy()

#%%

#feature engineering

#ONLY RUN THIS CELL ONCE 

#binary flag for when the singers sing
a = 1
for col in df.columns:
    sing_flag = df[col] > 0
    for i in range(len(sing_flag.iloc[:])):
        sing_flag.iloc[i] = int(sing_flag.iloc[i])
    df[f'sing_flag{a}'] = sing_flag
    a = a + 1

#num of singers singing
df['number singing'] = df[f'sing_flag1'] + df[f'sing_flag2'] + df[f'sing_flag3'] + df[f'sing_flag4']

#octave of each singer
df['octave0'] = 0
df['octave1'] = 0
df['octave2'] = 0
df['octave3'] = 0


for i in range(4):
    octave = np.zeros(len(df.iloc[:,0]))
    
    for j in range(len(df.iloc[:,0])):
        octave[j] = np.floor(df.iloc[j,i]/12) - 1
        
    df[f'octave{i}'] = octave
        

#note of each singer
df['note0'] = 0
df['note1'] = 0
df['note2'] = 0
df['note3'] = 0


for i in range(4):
    note = np.zeros(len(df.iloc[:,0]))
    
    for j in range(len(df.iloc[:,0])):
        note[j] = np.mod(df.iloc[j,i],12)
        
    df[f'note{i}'] = note

#%%

#for each individual voice, build regressor on all current features, and previous {lag} data points

# set lag
# take {lag} previous data points and add to row as features (but only from the {lag}th data point onwards)
# set target var as the singers note(0-120) and remove the other singers


lag = 1
df1 = df.iloc[lag:,:]


for i in tqdm(range(lag,lag + len(df1.iloc[:,0]))):
    
    
    last_lag_points = df.iloc[i-lag:i,:]
    last_lag_points_flat = last_lag_points.iloc[0,:]
    
    for j in range(1,lag):
         last_lag_points_flat = pd.concat([last_lag_points_flat,last_lag_points.iloc[j,:]],axis = 0)
         last_lag_points_flat.index = range(len(last_lag_points_flat))
         
    if i == lag:
        df2 = pd.concat([df.iloc[i,:4],last_lag_points_flat])
    else:
        temp_row = pd.concat([df.iloc[i,:4],last_lag_points_flat])
        df2 = pd.concat([df2,temp_row], axis = 1)


df2 = df2.transpose()


#%%


#lets start with predicting singer4

df_4 = df2.iloc[:,3:]

from catboost import CatBoostRegressor
# Initialize data

train_data4 = np.array(df_4.iloc[:,1:].astype('int'))



train_labels4 = np.array(df_4.iloc[:,0].astype('int'))

# Initialize CatBoostRegressor
model4 = CatBoostRegressor(iterations=500,
                          learning_rate=0.1,
                          depth=3)

# Fit model
model4.fit(train_data4, train_labels4)

pred4 = np.round(model4.predict(pd.DataFrame(df_4.iloc[-1,1:]).transpose().astype('int')),0)

#%%

#singer3

df_3 = pd.concat([df2.iloc[:,2], df2.iloc[:,4:]],axis = 1)

from catboost import CatBoostRegressor
# Initialize data

train_data3 = np.array(df_3.iloc[:,1:].astype('int'))



train_labels3 = np.array(df_3.iloc[:,0].astype('int'))

# Initialize CatBoostRegressor
model3 = CatBoostRegressor(iterations=500,
                          learning_rate=0.1,
                          depth=3)

# Fit model
model3.fit(train_data3, train_labels3)

pred3 = np.round(model3.predict(pd.DataFrame(df_3.iloc[-1,1:]).transpose().astype('int')),0)

#%%

#singer2

df_2 = pd.concat([df2.iloc[:,1], df2.iloc[:,4:]],axis = 1)

from catboost import CatBoostRegressor
# Initialize data

train_data2 = np.array(df_2.iloc[:,1:].astype('int'))



train_labels2 = np.array(df_2.iloc[:,0].astype('int'))

# Initialize CatBoostRegressor
model2 = CatBoostRegressor(iterations=500,
                          learning_rate=0.1,
                          depth=3)

# Fit model
model2.fit(train_data2, train_labels2)

pred2 = np.round(model2.predict(pd.DataFrame(df_3.iloc[-1,1:]).transpose().astype('int')),0)

#%%

#singer1

df_1 = pd.concat([df2.iloc[:,0], df2.iloc[:,4:]],axis = 1)

from catboost import CatBoostRegressor
# Initialize data

train_data1 = np.array(df_1.iloc[:,1:].astype('int'))



train_labels1 = np.array(df_1.iloc[:,0].astype('int'))

# Initialize CatBoostRegressor
model1 = CatBoostRegressor(iterations=500,
                          learning_rate=0.1,
                          depth=3)

# Fit model
model1.fit(train_data1, train_labels1)

pred1 = np.round(model1.predict(pd.DataFrame(df_3.iloc[-1,1:]).transpose().astype('int')),0)
#%%





def next_note(df_pred, lag):
    df_pred_0 = df_pred.copy()
    df_pred_1 = df_pred.copy()
        #feature engineering
    
    #ONLY RUN THIS CELL ONCE 
    
    #binary flag for when the singers sing
    a = 1
    for col in df_pred_1.columns:
        sing_flag = df_pred_1[col] > 0
        for i in range(len(sing_flag.iloc[:])):
            sing_flag.iloc[i] = int(sing_flag.iloc[i])
        
        df_pred_1[f'sing_flag{a}'] = sing_flag
        a = a + 1
    
    #num of singers singing
    df_pred_1['number singing'] = df_pred_1[f'sing_flag1'] + df_pred_1[f'sing_flag2'] + df_pred_1[f'sing_flag3'] + df_pred_1[f'sing_flag4']
    
    #octave of each singer
    df_pred_1['octave0'] = 0
    df_pred_1['octave1'] = 0
    df_pred_1['octave2'] = 0
    df_pred_1['octave3'] = 0
    
    
    for i in range(4):
        octave = np.zeros(len(df_pred_1.iloc[:,0]))
        
        for j in range(len(df_pred_1.iloc[:,0])):
            octave[j] = np.floor(df_pred_1.iloc[j,i]/12) - 1
            
        df_pred_1[f'octave{i}'] = octave
            
    
    #note of each singer
    df_pred_1['note0'] = 0
    df_pred_1['note1'] = 0
    df_pred_1['note2'] = 0
    df_pred_1['note3'] = 0
    
    
    for i in range(4):
        note = np.zeros(len(df_pred_1.iloc[:,0]))
        
        for j in range(len(df_pred_1.iloc[:,0])):
            note[j] = np.mod(df_pred_1.iloc[j,i],12)
            
        df_pred_1[f'note{i}'] = note
        
        
    df1 = df_pred_1.iloc[lag:,:]
    
    
  #  this only necessary for very last rows, sort it out
    
    for i in range(lag + len(df1.iloc[:,0]) - 2,lag + len(df1.iloc[:,0])):
        
        last_lag_points = df_pred_1.iloc[i-lag:i,:]
        last_lag_points_flat = last_lag_points.iloc[0,:]
        
        for j in range(1,lag):
             last_lag_points_flat = pd.concat([last_lag_points_flat,last_lag_points.iloc[j,:]],axis = 0)
             last_lag_points_flat.index = range(len(last_lag_points_flat))
             
        if i == lag + len(df1.iloc[:,0]) - 2:
            df2 = pd.concat([df_pred_1.iloc[i,:4],last_lag_points_flat])
        else:
            temp_row = pd.concat([df_pred_1.iloc[i,:4],last_lag_points_flat])
            df2 = pd.concat([df2,temp_row], axis = 1)
    
    
    df2 = df2.transpose()
        
    df_1 = pd.concat([df2.iloc[:,0], df2.iloc[:,4:]],axis = 1)
    df_2 = pd.concat([df2.iloc[:,1], df2.iloc[:,4:]],axis = 1)
    df_3 = pd.concat([df2.iloc[:,2], df2.iloc[:,4:]],axis = 1)
    df_4 = pd.concat([df2.iloc[:,3], df2.iloc[:,4:]],axis = 1)
    
    pred1 = np.round(model1.predict(pd.DataFrame(df_1.iloc[-1,1:]).transpose().astype('int')),0)
    pred2 = np.round(model2.predict(pd.DataFrame(df_2.iloc[-1,1:]).transpose().astype('int')),0)
    pred3 = np.round(model3.predict(pd.DataFrame(df_3.iloc[-1,1:]).transpose().astype('int')),0)
    pred4 = np.round(model4.predict(pd.DataFrame(df_4.iloc[-1,1:]).transpose().astype('int')),0)
    next_pred = pd.DataFrame([pred1[0],pred2[0],pred3[0],pred4[0]]).transpose()
    
    df_pred_0.columns = range(4)
    df_pred = pd.concat([df_pred_0.iloc[:,:],next_pred.iloc[:,:]],axis = 0)
    df_pred.columns = ['Singer1','Singer2','Singer3','Singer4']
    return df_pred


#%%

its = 50

df_pred = next_note(df0, lag)
for i in tqdm(range(its)):
    df_pred = next_note(df_pred,lag)

df_pred.index = range(len(df_pred))