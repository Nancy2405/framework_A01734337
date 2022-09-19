#!/usr/bin/env python
# coding: utf-8

# In[286]:


#Importando las librerías

import pandas as pd
import numpy as np
import math
import operator

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random
import warnings
warnings.filterwarnings("ignore")


# In[136]:


#Cargar los datos
#Es un dataset de kaggle de https://www.kaggle.com/code/ashydv/sales-prediction-simple-linear-regression/data.
#Contiene datos sobre la cantidad gastada en plataformas de marketing, y las ventas que se realizan. 
sales=pd.read_csv("https://raw.githubusercontent.com/Nancy2405/framework_A01734337/main/advertising.xls")


# In[137]:


sales


# In[138]:


sales.corr()


# In[139]:


sales.shape


# In[140]:


def intro():
    print("----------------------------------------------------------------------------------------\n")
    print("Modelo para predecir la cantidad de ventas con base en lo que se gasta en plataformas   \n")
    print("                                      de marketing                                      \n")
    print("                   Realizado por: Nancy Lesly Segura Cuanalo A01734337                  \n")
    print("----------------------------------------------------------------------------------------\n")


# In[204]:


def show_variables():
    print("Las variables presentes en este modelo son: \n")
    for i in range(3):
        print(i+1," -> ",sales.columns[i])
    print("----------------------------------------------------------------------------------------\n")


# In[205]:


def final():
    print("----------------------------------------------------------------------------------------\n")
    print("Fin de la ejecución")
    print("Gracias por utilizar este modelo predictivo")
    print("----------------------------------------------------------------------------------------\n")


# In[272]:


#La regresión lineal en lo que consiste es en asignar un vector de peso al vector de características, que mejor se ajuste, y para encontrar este se pueden usar distintos métodos, en este caso usé Gradient Descent

class LinearRegression() :
      
    def __init__( self, tasa_aprendizaje, iteraciones ) :
          
        self.tasa_aprendizaje = tasa_aprendizaje
          
        self.iteraciones = iteraciones
  
              
    def fit( self, X, Y ) :
          
       
          
        self.m, self.n = X.shape
          
       
          
        self.W = np.zeros( self.n )
          
        self.b = 0
          
        self.X = X
          
        self.Y = Y
          
          
        
                  
        for i in range( self.iteraciones ) :
              
            self.actualizar_pesos()
              
        return self
      
   
    def actualizar_pesos( self ) :
             
        Y_pred = self.predict( self.X )
          
        
      
        dW = - ( 2 * ( self.X.T ).dot( self.Y - Y_pred )  ) / self.m
       
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m 
          
       
        self.W = self.W - self.tasa_aprendizaje * dW
      
        self.b = self.b - self.tasa_aprendizaje * db
          
        return self
      
   
      
    def predict( self, X ) :
        return X.dot( self.W ) + self.b
    


# In[273]:


#Definiendo "X" y "y" 
X=sales.iloc[:,0]
X=X.values.reshape((200, 1))
y=sales.Sales.values
show_variables()
print("En este caso se tomará TV para hacer predicciones, ya que es variable que más correlacionada está con el target")


# In[274]:


#División del set de datos 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print("\nDivisión del set de datos exitosa: 80% train - 20% test\n")
print("----------------------------------------------------------------------------------------\n")


# In[275]:


#Creación del modelo
model = LinearRegression(iteraciones = 50, tasa_aprendizaje = 0.01)
print("Creación de modelo \"Regresión lineal\" exitosa \n(Iteaciones:50,tasa de aprendizaje:0.1)\n") 


# In[276]:


#Entrenamiento del modelo
model.fit(X_train, y_train)
print("Entrenamiento del modelo exitoso")


# In[277]:


#Hacer predicciones
predicciones=model.predict(X_test)
print("\n Predicciones realizadas con éxito ")
print("Total de predicciones: ",len(predicciones),"\n")
print("----------------------------------------------------------------------------------------\n")


# In[278]:


print(predicciones)


# In[279]:


#Calificar el modelo
mse_model=mean_squared_error(predicciones,y_test)
print("----------------------------------------------------------------------------------------\n")
print("MSE para el modelo: ",mse_model)
print("----------------------------------------------------------------------------------------\n")


# In[280]:


#Entrenar con todos los datos 

#Creación del modelo
model2= LinearRegression(iteraciones = 100, tasa_aprendizaje = 0.01)
#Entrenamiento del modelo
model2.fit(X,y)
print("Creación y entrenamiento de nuevo modelo ahora con todos los datos")
print("----------------------------------------------------------------------------------------\n")


# In[281]:


print("Introduzca la cantidad gastada en TV para hacer una predicción: \n")
var=float(input((str(1)+"-> : TV ")))


# In[282]:


data_predict=np.array(var)


# In[283]:


prediccion=model2.predict(data_predict)


# In[284]:


print("\n La predicción para esa cantidad gastada es de: ",prediccion,"\n")


# In[285]:


final()

