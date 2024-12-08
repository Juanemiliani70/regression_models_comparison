
import os 
import pandas as pd 
from sklearn import linear_model 
import matplotlib.pyplot as pl 
from sklearn.metrics import r2_score 

os.chdir("C:/Users/Eco") 
archivo = 'RangoMedioCentral.csv' 

h = pd.read_csv(archivo, sep=';') 


x = h["Xi"]
y = h["Yi"]

pl.scatter(x, y) 
pl.title("Reg Plot")
pl.show() 


lista = []
for i in range(len(x)):
    lista.append([x[i], 0])

mod = linear_model.LinearRegression() 
reg = mod.fit(lista, y) 

print("coeficiente: ", reg.coef_[0]) 
print("interseccion: ", reg.intercept_) 


pred = []
for i in range(len(x)):
    pred.append([i, 0])
    
yp = reg.predict(pred) 


r2 = r2_score(y, yp) 
print("R-cuadrado: ", r2)

cant_x = 35 
x_proy = [] 
for i in range(cant_x): 
    x_proy.append([i, 0]) 


y_proy = reg.predict(x_proy) 


ymin = min(y_proy)
ymax = max(y_proy)

pl.plot(range(len(x_proy)), y_proy, color='g') 
pl.scatter(x, y, color='r') 
pl.xlabel("Xi") 
pl.ylabel("Yi") 
pl.title("Datos y recta de regresion") 
pl.show() 


yt = [] 
err = [] 

for i in range(len(x)):
    b = yp[i]
    yt.append(b)
    err.append(y[i] - b)


pl.scatter(x, err, color = 'b') 
pl.xlabel("Xi") 
pl.ylabel("Yi")
pl.title("errores de la recta de regresion") 
pl.show() 


pl.hist(err, bins=10)
pl.xlabel("Err") 
pl.ylabel("Cantidad") 
pl.title("Histograma de errores") 
pl.show() 
