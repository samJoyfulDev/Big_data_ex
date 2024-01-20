import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris= sns.load_dataset("iris")
from sklearn.model_selection import train_test_split
iris.head()

X = iris.drop('species', axis = 1)
y = iris[['species']]

#separación para entrena.. y test

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=30)

#libreria algoritmo y entrenamos
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
#agregar prediccciones
from sklearn.metrics import classification_report, confusion_matrix
#predicciones
pred = knn.predict(X_test)
print(pred)
#generamos el reporte de métricas para analizar
#resultado

report = classification_report(y_test, pred)
tabla = confusion_matrix(y_test, pred)
print(report)
print(tabla)

#calculamos exactitud
print(13+10+18)
print(13+10+18+3+1)
print(41/45)
#calculamos la puntuación
knn.score(X_test, y_test)
#calculamos el scorea a valores de entrenamiento

knn.score(X_train, y_train)
#establecer un número de vecinos

vecinos =np.arange(1,20)
print(vecinos)

#crear 2 matrices vacias 
train_2 =np.empty(len(vecinos))
test_2 =np.empty(len(vecinos))
print(train_2)
print(test_2)
#generamos bucle para registrar datos en las matrices 
 #se genera un bubcle para registrar
for i, k in enumerate(vecinos):
     km = KNeighborsClassifier(n_neighbors=k)
     knn.fit(X_train, y_train)
     test_2[i] = knn.score(X_test, y_test)
     train_2[i] = knn.score(X_train, y_train)
    
print(train_2)
print (test_2)

#crear gráfico

plt.title('NUMERO DE VECINOS PROXIMOS KINN')
plt.plot(vecinos, test_2, label='Exactitud de Test')
plt.plot(vecinos, train_2, label='Exactitud de Train')
plt.legend()
plt.xlabel('Número de vecinos')
plt.ylabel('con exactitid')
plt.show()