# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 04:56:47 2022

@author: Samir
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as s
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#ubicar datos
activ=pd.read_csv("student_prediction.csv")

#Verificamos si hay datos faltantes
s.heatmap(activ.isnull())
#DATOS NUMERICOS

X_numerical = activ[['WORK','ACTIVITY','PARTNER','SALARY','TRANSPORT']]
#s.heatmap(X_numerical.corr(),annot=True)

# #GRUPO
X_cat = activ[['AGE','GENDER','HS_TYPE','SCHOLARSHIP']] 
onehotencoder=OneHotEncoder()
X_cat=onehotencoder.fit_transform(X_cat).toarray()

categorical_data=X_cat.shape
#print(categorical_data)  
X_cat=pd.DataFrame(X_cat)

X_numerical=X_numerical.reset_index()
X_all=pd.concat([X_cat,X_numerical],axis=1)


X=X_all.iloc[:,:-1].values
y=X_all.iloc[:,-1:].values


scaler=MinMaxScaler()
y=scaler.fit_transform(y) 

#Separar set de datos en entrenamientos y prueba


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4)

#Definiendo modelo

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100,activation='relu',input_shape=(18, )))
model.add(tf.keras.layers.Dense(units=100,activation='relu'))
model.add(tf.keras.layers.Dense(units=100,activation='relu'))
model.add(tf.keras.layers.Dense(units=100,activation='relu'))
model.add(tf.keras.layers.Dense(units=1,activation='linear'))

model.summary()

#Compilado 
model.compile(optimizer='Adam',loss='mean_squared_error')

#entrenamiento
epochs_hist=model.fit(X_train,y_train,epochs=20,batch_size=50,validation_split=0.4)
#prediccion
epochs_hist.history.keys()

#Grafico
plt.subplots()
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Progreso de entrenamiento de modelo')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss','Validation Loss'])


#Prediccion
y_predict=model.predict(X_test)
plt.subplots()
plt.plot(y_test,y_predict,'^',color='r')
plt.xlabel('Prediccion del modelo')
plt.ylabel('valores verdaderos')

#evaluar eficacia
valid_loss, valid_accuracy= model.evaluate(validation_split=0.4)