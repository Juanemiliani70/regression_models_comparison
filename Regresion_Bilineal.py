#Importación de librerías
import os
import pandas as pd
from sklearn import linear_model 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d 
from sklearn.metrics import r2_score 
import matplotlib.pyplot as pl


os.chdir("C:/Users/Eco")
h = pd.read_csv("Bilineal.csv", sep=";")
print(h)


h.shape 


x = h["Xi"]
y = h["Yi"] 
z = h["Zi"] 


fig = plt.figure(figsize = (10,7)) 
ax = plt.axes(projection = "3d") 
ax.view_init(30, 30) 
ax.scatter3D(x, y, z, color='green') 
plt.show() 


lista = [] 
for i in range(len(x)): 
    lista.append([x[i], y[i]]) 
print(lista) 



reg = linear_model.LinearRegression().fit(lista, z)


xx = []
yy = []
zz = []


xmin = min(x)
xmax = max(x)
ymin = min(y)
ymax = max(y)


for i in range(101):
    for j in range(101):
        vx = xmin + (xmax - xmin) * i / 100. 
        vy = ymin + (ymax - ymin) * j / 100. 
        xx.append(vx)
        yy.append(vy)
        
        zz.append(reg.intercept_ + reg.coef_[0] * vx + reg.coef_[1] * vy)
        


zmin = min(zz)
zmax = max(zz)

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d") 
ax.view_init(0, 45) 
ax.scatter3D(xx, yy, zz, color='red', marker='.')
ax.scatter3D(x, y, z, color='green') 

plt.show() 


zz = reg.predict(lista)
print(zz) 
r2 = r2_score(z, zz) 
print(r2)


yt = []
err = []

for i in range(len(z)): 
    b = zz[i] 
    yt.append(b)
    err.append(z[i] - b) 
pl.scatter(z, err, color='b') 

pl.xlabel("z") 
pl.ylabel("Err")
pl.title("Errores de la recta de regresión")

pl.show()

