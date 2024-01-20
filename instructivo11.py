# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 18:57:20 2022

@author: pc
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

train = pd.read_csv("train.csv")
train.head()

plt.figure.Figure(figsize=(20,10))
sns.heatmap(train.corr().abs(), annot=True)

corr = train.corr().abs()
corr_SP = corr.loc[:, ['SalePrice']]

sns.boxplot(x='OverallQual', y='SalePrice', data=train)

train_selec = train.loc[:,[
    "OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF",
    "1stFlrSF","FullBath","TotRmsAbvGrd","YearBuilt"]]
plt.figure.Figure(figsize=(20,10))
sns.heatmap(train_selec.corr().abs(), annot=True)

train_selec2 = train.loc[:,["OverallQual","GrLivArea","GarageCars"]]
sns.pairplot(train_selec2)

from sklearn.model_selection import train_test_split

X = train.loc[:,["OverallQual","GrLivArea","GarageCars"]]
Y = train.loc[:,["SalePrice"]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=33)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

print(lm.intercept_)

lm.coef_
print(str(lm.coef_))

predicciones = lm.predict(X_test)
print(predicciones)

DataFramePredicciones = pd.DataFrame(predicciones)
DataFramePredicciones.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)
df_unido = y_test.join(DataFramePredicciones)
print(df_unido)

#METRICAS
from sklearn import metrics
print('MAE', metrics.mean_absolute_error(y_test, predicciones))
print('MSE', metrics.mean_squared_error(y_test,predicciones))
print('RMSE', np.sqrt(metrics.mean_absolute_error(y_test, predicciones)))

sns.displot(train.loc[:,['SalePrice']])

from sklearn.metrics import mean_squared_log_error
print('Log RMSE', np.sqrt(metrics.mean_squared_log_error(y_test, predicciones)))

test = pd.read_csv('test.csv')
X = test.loc[:,["OverallQual","GrLivArea","GarageCars"]]

X.isna().sum()
X['GarageCars'].fillna(0,inplace = True)
predicciones = lm.predict(X)

#resultado matricial
predicciones = lm.predict(X)
DataFramePredicciones = pd.DataFrame(predicciones)
DataFramePredicciones.reset_index(drop = True, inplace = True)
X_test.reset_index(drop = True, inplace = True)
df_entrega = X.join(DataFramePredicciones)
print(df_entrega)


predicciones = lm.predict(X)
DataFramePredicciones = pd.DataFrame(predicciones)
DataFramePredicciones.reset_index(drop = True, inplace = True)
id = test.loc[:,['Id']]
id.reset_index(drop = True, inplace = True)
df_entrega = id.join(DataFramePredicciones)
print(df_entrega)

df_entrega.columns = ['Id','SalePrice']
df_entrega.to_csv('entrega.csv',index = False)

sns.displot(df_unido.loc[:,['SalePrice']], color="skyblue", label="X", kde=True)
