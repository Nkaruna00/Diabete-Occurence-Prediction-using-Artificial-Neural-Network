#!/usr/bin/env python3
# -*- coding: utf-8 -*-

 #Import the libraries


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

#Set the seed
from numpy.random import seed
seed(1)

sns.set()


# Load the Dataset
data = pd.read_csv("diabetes.csv",sep=',')


X = data.iloc[:,0:8].values
Y = data.iloc[:,[8]].values

"""
X = data[['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
Y = data['Outcome']
"""

#Split the dataset 80 / 20
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0,shuffle = False)
poids= 0.0005


# Scale the predictor variables
scaler = Scaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# Build the model

model = Sequential()
model.add(Dense(64,input_dim = 8))
model.add(Activation('relu'))
model.add(Dropout(0.2))



model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1)) 

model.add(Activation('sigmoid'))

# Plot the model and save it
plot_model(model, to_file='model.png')


# Compile the model with Adam Optimizer
model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])


# Train the model
history = model.fit(x_train,y_train, validation_data=(x_test, y_test),batch_size = 10, epochs=200,verbose=1)


#Evaluate the model
print(" Evaluate model")
scores = model.evaluate(x_test, y_test, verbose=1)

#Save the model
model.save("diabetes.h5")


#Plot the accuracy along the number of epoch
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



# Calculate model accuracy
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




