import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error


#########################################################################
#                    REGRESSION LINEAIRE MULTIPLE                       #
#########################################################################

'''
   Implémentez un modèle de régression multiple sur la base de données issue du fichier nommé 
   data/boston_house_prices.csv (sans utiliser des modèles prédéfinis de python).
'''

boston = pd.read_csv("../P06_analyse1/Data_Regression/boston_house_prices.csv", sep = ",")

# Visualisation des données
print(boston.shape)
print(boston.head(3))
#       CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD  TAX  PTRATIO       B  LSTAT  MEDV
# 0  0.00632  18.0   2.31     0  0.538  6.575  65.2  4.0900    1  296     15.3  396.90   4.98  24.0
# 1  0.02731   0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242     17.8  396.90   9.14  21.6
# 2  0.02729   0.0   7.07     0  0.469  7.185  61.1  4.9671    2  242     17.8  392.83   4.03  34.7

# y (target) : MEDV (dernière colonne) median value of owner-occupied homes in \$1000s.
# X (données) : CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT (13 col)

# calcul des correlation entre y et chacun des x :
for col in boston.columns :
	print(col, boston[col].corr(boston['MEDV']))

# CRIM -0.3883046085868113 ##
# ZN 0.36044534245054277 ##
# INDUS -0.4837251600283727
# CHAS 0.17526017719029854 ##
# NOX -0.4273207723732826 ##
# RM 0.6953599470715395
# AGE -0.3769545650045963 ##  # fait sauter le code
# DIS 0.24992873408590388 ##
# RAD -0.38162623063977763 ##
# TAX -0.468535933567767 ##  # fait sauter le code
# PTRATIO -0.5077866855375617
# B 0.33346081965706653 ##  #fait sauter le code
# LSTAT -0.7376627261740147
# MEDV 1.0

boston2 = boston[['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'RAD', 'DIS', 'PTRATIO',  'LSTAT', 'MEDV']]

y = boston2.iloc[:,-1:].values
x = boston2.iloc[:,:-1].values


print(x.shape, y.shape) # 

X = np.hstack((x, np.ones(y.shape)))

theta = np.random.randn(10,1)
print(theta)
m = boston2.shape[0]

print(X.shape, y.shape, theta.shape, m) # (506, 10) (506, 1) (10, 1) 506


# Création du modèle (model(X,theta)) 

def modelmult(X, theta):
	return X.dot(theta)
# Fonction du coût (fonction_cout(X,y,theta))

def fonction_cout(X, y, theta):
	return 1/(2*m) * np.sum((modelmult(X, theta) - y)**2)

# Le gradient (gradient(X,y,theta))

def gradient(X,y,theta):
	return 1/m * X.T.dot(modelmult(X, theta) - y)
print(gradient(X, y, theta))
# Descente du gradient (descente_gradient(X,y,theta,alpha,n_iterations))

def descente_gradient(X,y,theta,alpha,n_iterations):
	for i in range(n_iterations): 
		theta = theta - alpha*gradient(X, y, theta)
	return theta


theta1 = descente_gradient(X, y, theta, 0.001, 5000)
theta2 = descente_gradient(X, y, theta, 0.0005, 3300)
theta3 = descente_gradient(X, y, theta, 0.002, 1500)

print(theta1, fonction_cout(X, y, theta1)) # 13.196
print(theta2, fonction_cout(X, y, theta2)) # 14.115
print(theta3, fonction_cout(X, y, theta3)) # 1050815795310118.4

print(mean_squared_error(y, modelmult(X, theta1))) # 26.460
print(mean_squared_error(y, modelmult(X, theta2))) # 28.035
print(mean_squared_error(y, modelmult(X, theta3))) # 7594507303458618.0

