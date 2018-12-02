import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras import metrics
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

training_set = pd.read_csv('train.csv')
lbl = LabelEncoder()
lbl.fit(training_set['penalty'])
training_set['penalty']=lbl.transform(training_set['penalty'])
x0 = training_set.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13]]
y = training_set[['time']]
y1 = training_set[['time']].values
x2=x0.drop(['scale',  'flip_y', 'n_informative'], axis=1)
ss = preprocessing.MinMaxScaler()
x=x2
x.iloc[:, [  3, 4, 6, 7, 8, 9]]=ss.fit_transform(x2.iloc[:, [  3, 4, 6, 7, 8, 9]])

m = Sequential()
m.add(Dense(256,activation='relu', input_shape=(x.shape[1],)))

m.add(Dense(256,activation='relu'))
m.add(Dense(256,activation='relu'))
m.add(Dense(256,activation='relu'))
m.add(Dense(1,activation='relu'))


m.compile(loss='mean_squared_error',
          optimizer='Adam',
          metrics=[metrics.mse, metrics.categorical_accuracy])

m.fit(x,
      y,
      epochs=500, callbacks=[
        EarlyStopping(monitor='val_loss', patience=40),
        ModelCheckpoint(
            'best.model',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )],
      verbose=2,
      validation_split=0.1,shuffle=True,

      )
# Load the best model
m.load_weights("best.model")



predict9 = m.predict(x)

test = pd.read_csv('test.csv')
test['penalty']=lbl.transform(test['penalty'])
xt = test.drop(['scale',  'flip_y', 'n_informative','id',], axis=1)
xt2=xt
xt2 .iloc[:, [  3, 4, 6, 7, 8, 9]]= ss.transform(xt.iloc[:, [  3, 4, 6, 7, 8, 9]])
yt=m.predict(xt2)
result=pd.DataFrame()
result['id']=range(0,100)
result['time']=yt
result.to_csv("result.csv",index=False)

