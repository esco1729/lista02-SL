import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg as lg

from sklearn.decomposition import PCA

X = pd.read_csv('weather.csv', sep=',')
station = list(X.get('station'))
X = np.array(X[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']])

# Normalize and center X
mu = np.mean(X, axis=0)
std = np.std(X, axis=0)
Xc = (X - mu) / std

print(station)

U , S , V = lg.svd(Xc)

x  = list(range(12))
y1 = list(V[1,:])
y2 = list(V[2,:])

plt.plot(x, y1, label='C1')
plt.plot(x, y2, label='C2')
plt.xlabel('meses')
plt.ylabel('coeficiente')
plt.legend()
plt.show()

X2d = Xc @ V.T[:,:2]
print(X2d.shape)

# grafica con etiquetas de estation en forma de puntos X2d
plt.scatter(X2d[:, 0], X2d[:, 1])
for i, txt in enumerate(station):
    plt.annotate(txt, (X2d[i, 0], X2d[i, 1]))
plt.xlabel('C1')
plt.ylabel('C2')
plt.title('PCA Visualization')
plt.show()

R = np.array((
    [0,1],
    [-1,0]))
X2d_rotated = X2d @ R

plt.scatter(X2d_rotated[:, 0], X2d_rotated[:, 1])
for i, txt in enumerate(station):
    plt.annotate(txt, (X2d_rotated[i, 0], X2d_rotated[i, 1]))
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('PCA Visualization (Rotated)')
plt.show()
