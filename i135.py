import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
iris= sns.load_dataset("iris")
from sklearn.model_selection import train_test_split
X = iris.drop('species', axis=1)
y = iris[['species']]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=30)

from sklearn.tree import DecisionTreeClassifier
arbol = DecisionTreeClassifier()
arbol.fit(X_train, y_train)
#importamos
from sklearn import tree
tree.plot_tree(arbol)

X_nombre = list(X.columns)
classes = ['setosa','versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1, ncols=1,figsize = (3,3), dpi= 300)
tree.plot_tree(arbol,feature_names= X_nombre, class_names =classes, filled=True)
fig.savefig('imagen.png')
#CREAMOS LAS PREDICCIONES
pred = arbol.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

#arboles aleatorios

from sklearn.ensemble import RandomForestClassifier

#El número 20 es número de bosques aleatorios
rfc = RandomForestClassifier(n_estimators=20, random_state=33)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))