#Bach Project

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#import catboost
from tqdm import tqdm 

#%%

df = pd.read_csv("C:/Users/alex/Documents/myPython/RUG/Machine Learning 2020/F.txt", sep="\t", header=None)
df.columns = ['Singer1','Singer2','Singer3','Singer4']

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


lag = 5
df1 = df.iloc[5:,:]


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

df3 = df2.iloc[:,3:]









