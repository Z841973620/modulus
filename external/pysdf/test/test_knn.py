import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from pysdf.knn import find_knn

# def test_clusters():

x0 = 0.5
x1 = 1.0

N = 100

x = np.random.normal(x0, 0.1, N)
y = np.random.normal(x0, 0.1, N)
z = np.random.normal(x0, 0.1, N)

xyz = np.zeros(N * 3)
for i in range(N):
    xyz[3*i + 0] = x[i]
    xyz[3*i + 1] = y[i]
    xyz[3*i + 2] = z[i]

query_points = np.array([x0, x0, x0])
k = 10
query_results = find_knn(xyz, query_points, k)

x_query = np.zeros(k)
y_query = np.zeros(k)
z_query = np.zeros(k)

for i in range(k):
    x_query[i] = query_results[0][3*i+0]
    y_query[i] = query_results[0][3*i+1]
    z_query[i] = query_results[0][3*i+2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, "*")
ax.plot(x_query, y_query, z_query, "r*")
plt.show()

