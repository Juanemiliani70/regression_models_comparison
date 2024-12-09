
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pl 
from sklearn.metrics import r2_score 

os.chdir("C:/Users/Eco") 
archivo = 'RangoTotal.csv' 

h = pd.read_csv(archivo, sep=';') 

x = h["Xi"] 
y = h["Yi"] 

pl.scatter(x, y) 
pl.title("Reg Plot") 
pl.show() 


n = 3
z = np.polyfit(x, y, n) 


coef = z 
print(coef[0], coef[1], coef[2]) 


a1 = coef[2]
a2 = coef[1]
a3 = coef[0]


p = np.poly1d(z) 
x_pred = range(51) 
y_p = p(x_pred) 


r2 = r2_score(y, y_p)
print("R-cuadrado: ", r2)




ymin = min(y_p)
ymax = max(y_p)

xmin = min(x_pred)
xmax = max(x_pred)


pl.xlim(xmin, xmax)
pl.ylim(ymin, ymax)

pl.plot(x_pred, y_p, color='g') 
pl.scatter(x, y, color='r')
#Etiquetas para los ejes x e y.
pl.xlabel("Xi") 
pl.ylabel("Yi")
pl.title("Datos y recta de regresión") 
pl.show() 


yt = []
err = []
for i in range(len(x_pred)): 
    b = y_p[i]
    yt.append(b) 
    err.append(y[i] - b) 
    

ymin = min(err)
ymax = max(err)
pl.ylim(ymin, ymax) 
pl.scatter(x, err, color='b') 

pl.xlabel("Xi") 
pl.ylabel("Err")
pl.title("Errores de la recta de regresión") 
pl.show()


pl.hist(err, bins=10) 

pl.xlabel("Err") 
pl.ylabel("Cantidad")
pl.title("Histograma de errores") 
pl.show()




