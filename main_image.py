from PIL import Image, ImageOps
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering


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

dbscan = KMeans(n_clusters=11)
clusters = dbscan.fit_predict(points)


positions = np.where(clusters == 8)
prov = points[positions][:]

# plt.scatter(prov[:, 1], prov[:, 0], cmap='plasma')
# plt.show()

# get max and mins
y_max = prov[:, 0].max()
x_max = prov[:, 1].max()
y_min = prov[:, 0].min()
x_min = prov[:, 1].min()
area = (x_min, y_min, x_max, y_max)

img = Image.fromarray(data)
plt.imshow(img)
plt.show()
print(area)
img = img.crop(area)
img.show()

print(data.shape)
# print(area)
# print(x_max)
# print(image.size)
# image = data.crop(area)
# plt.imshow(data)
# plt.show()
# print(area)
# print(image.size)
# ola = gray.crop(area)

# ola.show()
