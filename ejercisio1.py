# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 08:24:02 2022

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
prod=pd.read_csv("rentas.csv")

# #Verificamos si hay datos faltantes
#s.heatmap(prod.isnull())

#Eliminar columnas
prod=prod.drop(labels=['instant'],axis=1)
#'instant' es el nombre de la columna

prod=prod.drop(labels=['casual','registered'],axis=1)
# #eliminacion de varias columnas

# #cambiar el formato de fecha
# #el nombre de la columna donde se encuentra la fechas es 'dteday'
# #el formato origen es mes-día-año ubicación de los datos.
prod.dteday=pd.to_datetime(prod.dteday, format='%m/%d/%Y')
#bajo esta modalidad sale la hora con 00:00:00

prod.dteday=pd.to_datetime(prod["dteday"].dt.strftime('%Y-%m-%d'))
prod.index=pd.DatetimeIndex(prod.dteday)
prod=prod.drop(labels=['dteday'],axis=1)


#Graficamos los datos cargados
#Buscamos la frecuencia y lo representamos con W
#W=frecuencia por semana-week(W)
#linewidth=>Grafico de lineas

# plt.subplots()
# prod['cnt'].asfreq('W').plot(linewidth=3)
# plt.title('Frecuencia de productos')
# plt.xlabel('Periodo')
# plt.ylabel('Renta')
# plt.show()

# plt.subplots()
# prod['cnt'].asfreq('M').plot(linewidth=3)
# plt.title('Uso por Mes (M)')
# plt.xlabel('Mes')
# plt.ylabel('Renta')
# plt.show()

# plt.subplots()
# prod['cnt'].asfreq('Q').plot(linewidth=3)
# plt.title('Uso por Cuatrimestre (M)')
# plt.xlabel('Cuatrimestre')
# plt.ylabel('Renta')
# plt.show()

#DATOS NUMERICOS

X_numerical = prod[['temp','hum','windspeed','cnt']]
#s.heatmap(X_numerical.corr(),annot=True)

# # # #GRUPO
X_cat = prod[['season','yr','mnth','holiday','weekday','workingday','weathersit']] 
onehotencoder=OneHotEncoder()
X_cat=onehotencoder.fit_transform(X_cat).toarray()

categorical_data=X_cat.shape
#print(categorical_data)  

X_cat=pd.DataFrame(X_cat)

X_numerical=X_numerical.reset_index()

X_all=pd.concat([X_cat,X_numerical],axis=1)
X_all=X_all.drop('dteday', axis=1)   

X=X_all.iloc[:,:-1].values
y=X_all.iloc[:,-1:].values


scaler=MinMaxScaler()
y=scaler.fit_transform(y) 


#Separar set de datos en entrenamientos y prueba


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#Definiendo modelo

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100,activation='relu',input_shape=(35, )))
model.add(tf.keras.layers.Dense(units=100,activation='relu'))
model.add(tf.keras.layers.Dense(units=100,activation='relu'))
model.add(tf.keras.layers.Dense(units=100,activation='relu'))
model.add(tf.keras.layers.Dense(units=1,activation='linear'))

model.summary()


#Compilado 
model.compile(optimizer='Adam',loss='mean_squared_error')

#entrenamiento
epochs_hist=model.fit(X_train,y_train,epochs=20,batch_size=50,validation_split=0.2)


#Prediccion
epochs_hist.history.keys()

#Grafico

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Progreso de entrenamiento de modelo')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss','Validation Loss'])

#Prediccion
y_predict=model.predict(X_test)
plt.plot(y_test,y_predict,"^",color='r')
plt.xlabel('Prediccion del modelo')
plt.ylabel('valores verdaderos') 
