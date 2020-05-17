#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:35:12 2019

@author: nithushan
"""


import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras import regularizers
from keras import optimizers


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as Scaler

from numpy.random import seed
seed(1)

sns.set()

data = pd.read_csv("diabetes.csv",sep=',')


X = data.iloc[:,0:8].values
Y = data.iloc[:,[8]].values

"""
X = data[['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
Y = data['Outcome']
"""

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0,shuffle = False)
poids= 0.0005

scaler = Scaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential()
model.add(Dense(64,input_dim = 8))
model.add(Activation('relu'))
model.add(Dropout(0.2))


#model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))
#model.add(Dense(40))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
"""
FAUT METTRE DENSE(1) A LA FIN , 8 CA NE MARCHE PAS A LA FIN
"""
model.add(Dense(1)) 

model.add(Activation('sigmoid'))

plot_model(model, to_file='model.png')

#adam =  optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

#adam = keras.optimizers.Adam( beta_1=0.9, beta_2=0.999, amsgrad=False)
#adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train,y_train, validation_data=(x_test, y_test),batch_size = 10, epochs=200,verbose=1)

print(" Evaluate model")
scores = model.evaluate(x_test, y_test, verbose=1)
model.save("diabetes.h5")


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()




print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




