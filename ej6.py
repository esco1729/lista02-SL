import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_blobs
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split


# Establecemos la semilla para reproducibilidad
np.random.seed(42707)

#############################################
# 1. Datos circulares concéntricos
#############################################
# Utilizamos make_circles de sklearn.
# factor=0.5 indica la escala del círculo interior.

X1,y1 = make_circles(n_samples=1_000, noise=0.05, factor=0.3, random_state=0)


#############################################
# 2. Datos de cuatro cuadrantes
#############################################
# Generamos cuatro grupos, cada uno centrado en un cuadrante distinto.
n_points = 250
q1 = np.random.randn(n_points, 2) * 0.3 + np.array([1, 1])
q2 = np.random.randn(n_points, 2) * 0.3 + np.array([-1, 1])
q3 = np.random.randn(n_points, 2) * 0.3 + np.array([-1, -1])
q4 = np.random.randn(n_points, 2) * 0.3 + np.array([1, -1])
X2 = np.vstack([q1, q2, q3, q4])
y2 = np.vstack([np.array([i % 2] * n_points) for i in range(4)]).ravel()



#############################################
# 3. Datos de dos nubes de puntos gaussianas
#############################################
# Utilizamos make_blobs para generar dos clusters.
centers = [[-1, -1], [1, 1]]
X3, y3 = make_blobs(n_samples=500, centers=centers, cluster_std=0.5, random_state=42)

#############################################
# 4. Datos en forma de espiral (doble espiral)
#############################################
def generate_double_spiral(n_points=500, noise=0.2):
    n_points_per_spiral = n_points // 2
    theta = np.linspace(0, 4 * np.pi, n_points_per_spiral)
    r = theta
    # Primera espiral
    x1 = r * np.cos(theta)
    y1 = r * np.sin(theta)
    spiral1 = np.column_stack([x1, y1])
    # Segunda espiral (rotada 180°)
    x2 = r * np.cos(theta + np.pi)
    y2 = r * np.sin(theta + np.pi)
    spiral2 = np.column_stack([x2, y2])
    # Añadimos ruido gaussiano
    spiral1 += noise * np.random.randn(n_points_per_spiral, 2)
    spiral2 += noise * np.random.randn(n_points_per_spiral, 2)
    return np.vstack([spiral1, spiral2]),np.vstack([np.array([i] *n_points_per_spiral) for i in range(2)]).ravel()

X4, y4 = generate_double_spiral(n_points=500, noise=0.2)


# Normalizar los datos
X1 = (X1 - np.mean(X1, axis=0)) / np.std(X1, axis=0)
X2 = (X2 - np.mean(X2, axis=0)) / np.std(X2, axis=0)
X3 = (X3 - np.mean(X3, axis=0)) / np.std(X3, axis=0)
X4 = (X4 - np.mean(X4, axis=0)) / np.std(X4, axis=0)

print(y1.shape)
print(y2.shape)
print(y3.shape)
print(y4.shape)

# Plotea X1
plt.figure(figsize=(5, 5))
plt.scatter(X1[:, 0], X1[:, 1], c=y1)
plt.title('Datos circulares concéntricos')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Plotea X2
plt.figure(figsize=(5, 5))
plt.scatter(X2[:, 0], X2[:, 1], c=y2)
plt.title('Datos de cuatro cuadrantes')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Plotea X3
plt.figure(figsize=(5, 5))
plt.scatter(X3[:, 0], X3[:, 1], c= y3)
plt.title('Datos de dos nubes de puntos gaussianas')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Plotea X4
plt.figure(figsize=(5, 5))
plt.scatter(X4[:, 0], X4[:, 1], c= y4)
plt.title('Datos en forma de espiral (doble espiral)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Kernel PCA X1

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, stratify=y1, random_state=0)


Kpca1 = KernelPCA( n_components=None, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1)


X1_test_Kpca = Kpca1.fit(X1_train).transform(X1_test)


plt.figure(figsize=(5, 5))
plt.scatter(X1_test_Kpca[:, 1], X1_test_Kpca[:, 0], c=y1_test)
plt.title('Datos circulares concéntricos')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#  Kernel PCA X2

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, stratify=y2, random_state=0)

Kpca2 = KernelPCA( n_components=None, kernel="poly",degree=4, fit_inverse_transform=True, alpha=0.1)


X2_test_Kpca = Kpca2.fit(X2_train).transform(X2_test)


plt.figure(figsize=(5, 5))
plt.scatter(X2_test_Kpca[:, 1], X2_test_Kpca[:, 0], c=y2_test)
plt.title('Datos de cuatro cuadrantes')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Kernel PCA 3
X1_train, X1_test, y1_train, y1_test = train_test_split(X3, y3, stratify=y3, random_state=0)

#aplicar_kernel_pca(X3, y3, kernel="linear", title="Nubes de Puntos Gaussianas")
Kpca1 = KernelPCA( n_components=None, kernel="linear", gamma=11, fit_inverse_transform=True, alpha=0.1)


X1_test_Kpca = Kpca1.fit(X1_train).transform(X1_test)


plt.figure(figsize=(5, 5))
plt.scatter(X1_test_Kpca[:, 1], X1_test_Kpca[:, 0], c=y1_test)
plt.title('nubes de puntos gaussianas')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Kernel PCA 4
X1_train, X1_test, y1_train, y1_test = train_test_split(X4, y4, stratify=y4, random_state=0)


Kpca1 = KernelPCA( n_components=None, kernel="rbf", gamma=11, fit_inverse_transform=True, alpha=0.1)


X1_test_Kpca = Kpca1.fit(X1_train).transform(X1_test)


plt.figure(figsize=(5, 5))
plt.scatter(X1_test_Kpca[:, 1], X1_test_Kpca[:, 0], c=y1_test)
plt.title('Datos en forma de espiral')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
