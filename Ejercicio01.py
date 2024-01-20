import numpy as np
import datetime
import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(-1,28*28)
x_train.shape

x_test = x_test.reshape(-1,28*28)
x_test.shape

#Creando la red neuronal
#Creando modelo

model=tf.keras.models.Sequential()

#Agrendando la capa 1

model.add(tf.keras.layers.Dense(units=128,activation='relu',input_shape=(784,)))

#128 es la cantidad de neuronas de la capa
#relu es una funcion lineal rectificante
#28x28=784pixeles
#Agregando la capa 2
model.add(tf.keras.layers.Dense(units=64,activation='relu'))

#Capa DropOut (Desercion)
#La deserción es útil para regularizar modelos.
#Los elementos de entrada se establecen aleatoriamente en cero
#(y los otros elementos se reescalan)
#Cada nodo se convierte en independiente útil, y no confia 
#en la salida de otros nodos
#Método para combatir el sobreajuste en redes neuronales
#Dopout aproxima un numero exponencial de modelos para combinarlos
# y predecir la salida

model.add(tf.keras.layers.Dropout(0.2))

#Valor aleatorio 0.2 se puede cambiar de 0 a 1

#Capa Output => Capa de salida

model.add(tf.keras.layers.Dense(units=10,))
#10 porque son salidas psoibles=>
#Seccion labels=>Aparece la lista de respuestas
#Softmax se usa cuando queremos construir un clasificador de clases múltiples
#que resuelve el problema de asignar una instancia a una clase
#Cuando el número de clases posibles es mayor que dos 

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
model.summary()
#sparse_cateorial_crossentropy =>Entropia cruzada / Uso como funcion de perdida.

model.fit (x_train, y_train, epochs=10)

#Evaluacion

test_loss, test_accuracy=model.evaluate(x_test, y_test)
print('Test Accuracy: {}'.format(test_accuracy))



