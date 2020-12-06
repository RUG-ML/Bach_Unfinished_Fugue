#Bach Project

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xgboost

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


