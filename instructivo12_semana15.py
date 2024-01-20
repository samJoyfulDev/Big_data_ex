import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#IRIS ES UN DATAFRAME DE EJEMPLO
iris = sns.load_dataset("iris")

iris.head()
#Si se desea filtrar los datos se puede realizar
iris_v = iris[iris['species']!='setosa']

sns.pairplot(iris_v, hue = 'species')

#ELIMINAMOS LA COLUMNA SPECIES
X = iris_v.drop('species', axis = 1)
Y = iris_v['species']

#PREPARAMOS MODELO CON DATOS SEPARADOS
#DEL PASO ANTERIOR

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.30, random_state=100)

#entrenamos el modelo con datos separados
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
#INICIAMOS LOS TEST PARA PREDICCIONES
predicciones = logmodel.predict(X_test)
print(predicciones)
#analizamos metricas
from sklearn.metrics import classification_report
print(classification_report(y_test, predicciones))
#mostramos tabla de confusión
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predicciones)

#ahora vamos a obteer curva ROC
from sklearn.metrics import roc_curve
y_pred_prob = logmodel.predict_proba(X_test)[:,1]

#SE DEBE ESCOGER una de las columnas
#porque los valores son 0 y 1
#0 es una columna y 1 otra columna
roc_curve(y_test, y_pred_prob,'virginica')

#obtener falso positivo
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob,'virginica')
print(fpr)
print(tpr)
print(threshold)

#se dibuja

plt.plot(fpr, tpr, color='red', label = 'Curva ROC')
plt.plot([0,1][0,1], color = 'blue', linestyle = '--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC')
plt.show()
