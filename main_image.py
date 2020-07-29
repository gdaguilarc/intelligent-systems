from PIL import Image, ImageOps
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering


new_img = [[255 for _ in range(28)] for _ in range(28)]
for i in range(28):
    new_img[i][i] = 0

 # mg = Image.fromarray(np.array(new_img))
# img.convert("L").show()


image = Image.open("numeros.jpeg")
gray = ImageOps.grayscale(image)
data = np.asarray(gray)

points = []
maxX = len(data)
maxY = len(data[0])
for i in range(maxX):
    for j in range(maxY):
        if data[i][j] < 125:
            points.append((i, j))

points = np.array(points)

dbscan = DBSCAN(eps=3.8, min_samples=1)
clusters = dbscan.fit_predict(points)
plt.scatter(points[:, 1], points[:, 0], c=clusters, cmap='plasma')

plt.show()
