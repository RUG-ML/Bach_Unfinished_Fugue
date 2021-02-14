import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import catboost
from tqdm import tqdm 
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor

#%%

df = pd.read_csv("/Users/alexhill/Documents/myPython/RUG/Machine Learning/F.txt", sep="\t", header=None)
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

#Analysing the lengths of notes and finding their frequencies

lengths1 = []
first_val1 = df.iloc[0,0]
length1 = 0

lengths2 = []
first_val2= df.iloc[0,1]
length2 = 0

lengths3 = []
first_val3 = df.iloc[0,2]
length3 = 0

lengths4 = []
first_val4 = df.iloc[0,3]
length4 = 0

for i in range(1,len(df.iloc[:,0])):
    if df.iloc[i,0] == df.iloc[i-1,0]:
        length1 = length1 + 1
    else:
        lengths1.append(length1)
        length1= 0
        
    if df.iloc[i,1] == df.iloc[i-1,1]:
        length2 = length2 + 1
    else:
        lengths2.append(length2)
        length2 = 0
        
    if df.iloc[i,2] == df.iloc[i-1,2]:
        length3 = length3 + 1
    else:
        lengths3.append(length3)
        length3 = 0
        
    if df.iloc[i,3] == df.iloc[i-1,3]:
        length4 = length4 + 1
    else:
        lengths4.append(length4)
        length4 = 0
      
for i in range(len(lengths1)-6):
    if lengths1[i] > 50:
        lengths1.pop(i)

for i in range(len(lengths2)-3):
    if lengths2[i] > 50:
        lengths2.pop(i)
        
for i in range(len(lengths3)-5):
    if lengths3[i] > 50:
        lengths3.pop(i)
        
for i in range(len(lengths4)-5):
    if lengths4[i] > 50:
        lengths4.pop(i)
        
length_pdf1 = np.zeros(51)
length_pdf2 = np.zeros(51)
length_pdf3 = np.zeros(51)
length_pdf4 = np.zeros(51)

for i in range(len(lengths1)):
    length_pdf1[lengths1[i]] = length_pdf1[lengths1[i]] + 1

for i in range(len(lengths2)):
    length_pdf2[lengths2[i]] = length_pdf2[lengths2[i]] + 1
    
for i in range(len(lengths3)):
    length_pdf3[lengths3[i]] = length_pdf3[lengths3[i]] + 1
    
for i in range(len(lengths4)):
    length_pdf4[lengths4[i]] = length_pdf4[lengths4[i]] + 1
    
length_choices = np.array(range(51))
    
length_pdf1 = length_pdf1/length_pdf1.sum()
length_pdf2 = length_pdf2/length_pdf2.sum()
length_pdf3 = length_pdf3/length_pdf3.sum()
length_pdf4 = length_pdf4/length_pdf4.sum()

#%%

plt.figure()

plt.hist(lengths1, bins = 30)
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.show()


#%%

#for each individual voice, build regressor on all current features, and previous {lag} data points

# set lag
# take {lag} previous data points and add to row as features (but only from the {lag}th data point onwards)
# set target var as the singers note(0-120) and remove the other singers


lag = 50
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


#singer4

df_4 = df2.iloc[:,3:]


# Initialize data

train_data4 = np.array(df_4.iloc[:,1:].astype('int'))

train_labels4 = np.array(df_4.iloc[:,0].astype('int'))


splitter = TimeSeriesSplit(n_splits=5)
its_list = [10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,550,600]#,650,700,750,800,850,900,950,1000]
depth_list = [2,3,4,5,6,7,8,9,10]

av_train_errors = []
av_test_errors = []

for its_num in tqdm(its_list):
    train_errors = []
    test_errors = []
    for train_index, test_index in splitter.split(train_data4):
        
        X_train, X_test = train_data4[train_index], train_data4[test_index]
        y_train, y_test = train_labels4[train_index], train_labels4[test_index]
        
    
        # Initialize CatBoostRegressor
        model4 = CatBoostRegressor(iterations=its_num,
                                  learning_rate=0.01,
                                  depth=3)
    
        # Fit model
        model4.fit(X_train, y_train)
        
        pred_train4 = np.round(model4.predict(pd.DataFrame(X_train).astype('int')),0)
        pred_test4 = np.round(model4.predict(pd.DataFrame(X_test).astype('int')),0)
        
        train_error = ((y_train-pred_train4)**2).mean()
        train_errors.append(train_error)
        
        test_error = ((y_test-pred_test4)**2).mean()
        test_errors.append(test_error)
        
    av_train_error = np.array(train_errors).mean()
    av_train_errors.append(av_train_error)
    
    av_test_error = np.array(test_errors).mean()
    av_test_errors.append(av_test_error)
    
av_test_errors4 = av_test_errors
av_train_errors4 = av_train_errors
    
    
#%%


plt.figure(figsize = (8,5))


plt.plot(its_list,av_train_errors4, label = ' Train error (MSE)', color = 'blue', linestyle = '-', alpha = 0.7)
plt.plot(its_list,av_test_errors4, label = ' Test error (MSE)', color =  'g', linestyle = '-', alpha = 0.7)

plt.scatter(its_list,av_train_errors4, s = 10, marker = 'x', color = 'blue')
plt.scatter(its_list,av_test_errors4, s = 10, marker = 'x', color = 'g')

plt.legend()
plt.grid(alpha = 0.5, linestyle = '--')
plt.xlabel('Number of Trees')
plt.ylabel('MSE')

plt.title('Singer 4')
plt.show()


#%%


#singer3

df_3 = pd.concat([df2.iloc[:,2], df2.iloc[:,4:]],axis = 1)


# Initialize data

train_data3 = np.array(df_3.iloc[:,1:].astype('int'))

train_labels3 = np.array(df_3.iloc[:,0].astype('int'))


splitter = TimeSeriesSplit(n_splits=5)
its_list = [10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,550,600]#,650,700,750,800,850,900,950,1000]
depth_list = [2,3,4,5,6,7,8,9,10]

av_train_errors = []
av_test_errors = []

for its_num in tqdm(its_list):
    train_errors = []
    test_errors = []
    for train_index, test_index in splitter.split(train_data4):
        
        X_train, X_test = train_data3[train_index], train_data3[test_index]
        y_train, y_test = train_labels3[train_index], train_labels3[test_index]
        
    
        # Initialize CatBoostRegressor
        model3 = CatBoostRegressor(iterations=its_num,
                                  learning_rate=0.01,
                                  depth=3)
    
        # Fit model
        model3.fit(X_train, y_train)
        
        pred_train3 = np.round(model3.predict(pd.DataFrame(X_train).astype('int')),0)
        pred_test3 = np.round(model3.predict(pd.DataFrame(X_test).astype('int')),0)
        
        train_error = ((y_train-pred_train3)**2).mean()
        train_errors.append(train_error)
        
        test_error = ((y_test-pred_test3)**2).mean()
        test_errors.append(test_error)
        
    av_train_error = np.array(train_errors).mean()
    av_train_errors.append(av_train_error)
    
    av_test_error = np.array(test_errors).mean()
    av_test_errors.append(av_test_error)
    
av_test_errors3 = av_test_errors
av_train_errors3 = av_train_errors
    
    
#%%


plt.figure(figsize = (8,5))


plt.plot(its_list,av_train_errors3, label = ' Train error (MSE)', color = 'r', linestyle = '-', alpha = 0.7)
plt.plot(its_list,av_test_errors3, label = ' Test error (MSE)', color =  'g', linestyle = '-', alpha = 0.7)

plt.scatter(its_list,av_train_errors3, s = 10, marker = 'x', color = 'r')
plt.scatter(its_list,av_test_errors3, s = 10, marker = 'x', color = 'g')

plt.legend()
plt.grid(alpha = 0.5, linestyle = '--')
plt.xlabel('Number of Trees')
plt.ylabel('MSE')

plt.title('Singer 3')
plt.show()


#%%

#singer2

df_2 = pd.concat([df2.iloc[:,1], df2.iloc[:,4:]],axis = 1)


# Initialize data

train_data2 = np.array(df_2.iloc[:,1:].astype('int'))

train_labels2 = np.array(df_2.iloc[:,0].astype('int'))


splitter = TimeSeriesSplit(n_splits=5)
its_list = [10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,550,600]#,650,700,750,800,850,900,950,1000]
depth_list = [2,3,4,5,6,7,8,9,10]

av_train_errors = []
av_test_errors = []

for its_num in tqdm(its_list):
    train_errors = []
    test_errors = []
    for train_index, test_index in splitter.split(train_data4):
        
        X_train, X_test = train_data2[train_index], train_data2[test_index]
        y_train, y_test = train_labels2[train_index], train_labels2[test_index]
        
    
        # Initialize CatBoostRegressor
        model2 = CatBoostRegressor(iterations=its_num,
                                  learning_rate=0.01,
                                  depth=3)
    
        # Fit model
        model2.fit(X_train, y_train)
        
        pred_train2 = np.round(model2.predict(pd.DataFrame(X_train).astype('int')),0)
        pred_test2 = np.round(model2.predict(pd.DataFrame(X_test).astype('int')),0)
        
        train_error = ((y_train-pred_train2)**2).mean()
        train_errors.append(train_error)
        
        test_error = ((y_test-pred_test2)**2).mean()
        test_errors.append(test_error)
        
    av_train_error = np.array(train_errors).mean()
    av_train_errors.append(av_train_error)
    
    av_test_error = np.array(test_errors).mean()
    av_test_errors.append(av_test_error)
    
av_test_errors2 = av_test_errors
av_train_errors2 = av_train_errors
    
    
#%%


plt.figure(figsize = (8,5))


plt.plot(its_list,av_train_errors2, label = ' Train error (MSE)', color = 'r', linestyle = '-', alpha = 0.7)
plt.plot(its_list,av_test_errors2, label = ' Test error (MSE)', color =  'g', linestyle = '-', alpha = 0.7)

plt.scatter(its_list,av_train_errors2, s = 10, marker = 'x', color = 'r')
plt.scatter(its_list,av_test_errors2, s = 10, marker = 'x', color = 'g')

plt.legend()
plt.grid(alpha = 0.5, linestyle = '--')
plt.xlabel('Number of Trees')
plt.ylabel('MSE')

plt.title('Singer 2')
plt.show()



#%%

#singer1

df_1 = pd.concat([df2.iloc[:,0], df2.iloc[:,4:]],axis = 1)


# Initialize data

train_data1 = np.array(df_1.iloc[:,1:].astype('int'))

train_labels1 = np.array(df_1.iloc[:,0].astype('int'))


splitter = TimeSeriesSplit(n_splits=5)
its_list = [10,20,30,40,50,60,70,80,90,100,150]#,200,250,300,350,400,450,500,550,600]#,650,700,750,800,850,900,950,1000]
depth_list = [2,3,4,5,6,7,8,9,10]

av_train_errors = []
av_test_errors = []

for its_num in tqdm(its_list):
    train_errors = []
    test_errors = []
    for train_index, test_index in splitter.split(train_data4):
        
        X_train, X_test = train_data1[train_index], train_data1[test_index]
        y_train, y_test = train_labels1[train_index], train_labels1[test_index]
        
    
        # Initialize CatBoostRegressor
        model1 = CatBoostRegressor(iterations=its_num,
                                  learning_rate=0.1,
                                  depth=3)
    
        # Fit model
        model1.fit(X_train, y_train)
        
        pred_train1 = np.round(model1.predict(pd.DataFrame(X_train).astype('int')),0)
        pred_test1 = np.round(model1.predict(pd.DataFrame(X_test).astype('int')),0)
        
        train_error = ((y_train-pred_train1)**2).mean()
        train_errors.append(train_error)
        
        test_error = ((y_test-pred_test1)**2).mean()
        test_errors.append(test_error)
        
    av_train_error = np.array(train_errors).mean()
    av_train_errors.append(av_train_error)
    
    av_test_error = np.array(test_errors).mean()
    av_test_errors.append(av_test_error)
    
av_test_errors1 = av_test_errors
av_train_errors1 = av_train_errors
    
    
#%%


plt.figure(figsize = (8,5))


plt.plot(its_list,av_train_errors1, label = ' Train error (MSE)', color = 'r', linestyle = '-', alpha = 0.7)
plt.plot(its_list,av_test_errors1, label = ' Test error (MSE)', color =  'g', linestyle = '-', alpha = 0.7)

plt.scatter(its_list,av_train_errors1, s = 10, marker = 'x', color = 'r')
plt.scatter(its_list,av_test_errors1, s = 10, marker = 'x', color = 'g')

plt.legend()
plt.grid(alpha = 0.5, linestyle = '--')
plt.xlabel('Number of Trees')
plt.ylabel('MSE')

plt.title('Singer 1')
plt.show()




#%%



plt.figure(figsize = (8,5))


plt.plot(its_list,av_train_errors1, label = ' Train error (MSE)', color = 'blue', linestyle = '-', alpha = 0.5)
plt.plot(its_list,av_test_errors1, label = ' Test error (MSE)', color =  'orange', linestyle = '-', alpha = 1)

plt.scatter(its_list,av_train_errors1, s = 10, marker = 'x', color = 'blue', alpha = 0.6)
plt.scatter(its_list,av_test_errors1, s = 10, marker = 'x', color = 'orange', alpha = 0.6)

plt.plot(its_list,av_train_errors2, color = 'blue', linestyle = '-', alpha = 0.5)
plt.plot(its_list,av_test_errors2,  color =  'orange', linestyle = '-', alpha = 1)

plt.scatter(its_list,av_train_errors2, s = 10, marker = 'x', color = 'blue', alpha = 0.6)
plt.scatter(its_list,av_test_errors2, s = 10, marker = 'x', color = 'orange', alpha = 0.6)

plt.plot(its_list,av_train_errors3,  color = 'blue', linestyle = '-', alpha = 0.5)
plt.plot(its_list,av_test_errors3,  color =  'orange', linestyle = '-', alpha = 1)

plt.scatter(its_list,av_train_errors3, s = 10, marker = 'x', color = 'blue', alpha = 0.6)
plt.scatter(its_list,av_test_errors3, s = 10, marker = 'x', color = 'orange', alpha = 0.6)

plt.plot(its_list,av_train_errors4,  color = 'blue', linestyle = '-', alpha = 0.5)
plt.plot(its_list,av_test_errors4, color =  'orange', linestyle = '-', alpha = 1)

plt.scatter(its_list,av_train_errors4, s = 10, marker = 'x', color = 'blue', alpha = 0.6)
plt.scatter(its_list,av_test_errors4, s = 10, marker = 'x', color = 'orange', alpha = 0.6)
plt.legend(prop={'size': 12})
plt.grid(alpha = 0.5, linestyle = '--')

plt.xlabel('Number of Trees')
plt.ylabel('MSE')

plt.title('Singers 1-4 : MSE vs Number of Trees')
plt.show()



#%%


def next_note(df_pred, lag):
    df_pred_0 = df_pred.copy()
    df_pred_1 = df_pred.copy()
        #feature engineering
          
    global change_note1
    global change_note2
    global change_note3
    global change_note4
    global length_pdf1
    global length_pdf2
    global length_pdf3
    global length_pdf4
    
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
    
    
    if change_note1 == 0:
        pred1 = np.round(model1.predict(pd.DataFrame(df_1.iloc[-1,1:]).transpose().astype('int')),0)
        change_note1 = np.random.choice(length_choices, size  = 1, p = length_pdf1)
        pred1 = pred1[0]
        
        if pred1 == df_pred.iloc[-1,0]:
            if pred1 <  15:
                pred1 = pred1 + float(round(np.random.normal(45, 10, 1)[0],0))
            else:
                pred1 = pred1 + float(round(np.random.normal(0, 6, 1)[0],0))
        
        if pred1 < 0:
                pred1 = float(0)
    else:
        change_note1 = change_note1 - 1
        pred1 = df_pred.iloc[-1,0]
        
    
    if change_note2 == 0:
        pred2 = np.round(model2.predict(pd.DataFrame(df_2.iloc[-1,1:]).transpose().astype('int')),0)
        change_note2 = np.random.choice(length_choices, size  = 1, p = length_pdf2)
        pred2 = pred2[0]
        
        if pred2 == df_pred.iloc[-1,1]:
            if pred2 < 15:
                pred2 = pred2 + float(round(np.random.normal(45, 10, 1)[0],0))
            else:
                pred2 = pred2 + float(round(np.random.normal(0, 6, 1)[0],0))
        
        if pred2 < 0:
                pred2 = float(0)        
    else:
        change_note2 = change_note2 - 1
        pred2 = df_pred.iloc[-1,1]
        
        
    if change_note3 == 0:
        pred3 = np.round(model3.predict(pd.DataFrame(df_3.iloc[-1,1:]).transpose().astype('int')),0)
        change_note3 = np.random.choice(length_choices, size  = 1, p = length_pdf3)
        pred3 = pred3[0]
        
        if pred3 == df_pred.iloc[-1,2]:
            if pred3 < 15:
                pred3 = pred3 + float(round(np.random.normal(45, 10, 1)[0],0))
            else:
                pred3 = pred3 + float(round(np.random.normal(0, 6, 1)[0],0))
        
        if pred3 < 0:
                pred3 = float(0)
    else:
        change_note3 = change_note3 - 1
        pred3 = df_pred.iloc[-1,2]
        
    if change_note4 == 0:
        pred4 = np.round(model4.predict(pd.DataFrame(df_4.iloc[-1,1:]).transpose().astype('int')),0)
        change_note4 = np.random.choice(length_choices, size  = 1, p = length_pdf4)
        pred4 = pred4[0]
        
        if pred4 == df_pred.iloc[-1,3]:
            if pred4 < 15:
                pred4 = pred4 + float(round(np.random.normal(45, 10, 1)[0],0))
            else:
                pred4 = pred4 + float(round(np.random.normal(0, 6, 1)[0],0))
        
        if pred4 < 0:
                pred4 = float(0)
    else:
        change_note4 = change_note4 - 1
        pred4 = df_pred.iloc[-1,3]
    
    
    next_pred = pd.DataFrame([pred1,pred2,pred3,pred4]).transpose()
    
    df_pred_0.columns = range(4)
    df_pred = pd.concat([df_pred_0.iloc[:,:],next_pred.iloc[:,:]],axis = 0)
    df_pred.columns = ['Singer1','Singer2','Singer3','Singer4']
    
    return df_pred


#%%

its = 100

change_note1 = 0
change_note2 = 0
change_note3 = 0
change_note4 = 0


df_pred = next_note(df0, lag)
for i in tqdm(range(its)):
    df_pred = next_note(df_pred,lag)

df_pred.index = range(len(df_pred))

#%%

#removing notes that are very low

df_pred_post = df_pred.copy()
for i in range(3823,len(df_pred_post.iloc[:,0])):
    for j in range(4):
        if df_pred_post.iloc[i,j] < 20:
            df_pred_post.iloc[i,j] = 0
            
            

#%%

np.savetxt('bach_pred_same_length100.txt', df_pred_post.values, fmt='%d', delimiter = '\t')


