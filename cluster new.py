# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
	
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
%matplotlib inline
# Importing Dataset
from sklearn import datasets
iris = datasets.load_iris()
iris_data = pd.DataFrame(iris.data)
iris_data.columns = iris.feature_names
iris_data['Type']=iris.target
iris_data.head()
# Preparing Data
iris_X = iris_data.iloc[:, [0, 1, 2,3]].values


iris_Y = iris_data['Type']
iris_Y1 = np.array(iris_Y)

# Visualise Classes


plt.scatter(iris_X[iris_Y == 0, 0], iris_X[iris_Y1 == 0, 1], s = 80, c = 'orange', label = 'Iris-setosa')
plt.scatter(iris_X[iris_Y == 1, 0], iris_X[iris_Y1 == 1, 1], s = 80, c = 'yellow', label = 'Iris-versicolour')
plt.scatter(iris_X[iris_Y == 2, 0], iris_X[iris_Y1 == 2, 1], s = 80, c = 'green', label = 'Iris-virginica')
plt.legend()

#Deciding Value of K
wcss=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(iris_X)
    wcss.append(kmeans.inertia_)

#WCSS or within-cluster sum of squares is a measure of how internally coherent the clusters are
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
# Running K-Means Model

cluster_Kmeans = KMeans(n_clusters=3)
model_kmeans = cluster_Kmeans.fit(iris_X)
pred_kmeans = model_kmeans.labels_
pred_kmeans

# Visualizing Output

plt.scatter(iris_X[pred_kmeans == 0, 0], iris_X[pred_kmeans == 0, 1], s = 80, c = 'orange', label = 'Iris-setosa')
plt.scatter(iris_X[pred_kmeans == 1, 0], iris_X[pred_kmeans == 1, 1], s = 80, c = 'yellow', label = 'Iris-versicolour')
plt.scatter(iris_X[pred_kmeans == 2, 0], iris_X[pred_kmeans == 2, 1], s = 80, c = 'green', label = 'Iris-virginica')
plt.legend()

# Hierarchical Clustering
iris_X = iris_data[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]


Z = sch.linkage(iris_X, method = 'median')
plt.figure(figsize=(20,7))
den = sch.dendrogram(Z)
plt.title('Dendrogram for the clustering of the dataset iris)')
plt.xlabel('Type')
plt.ylabel('Euclidean distance in the space with other variables')

	
cluster_H = AgglomerativeClustering(n_clusters=3)
# Fitting Model

model_clt = cluster_H.fit(iris_X)
model_clt
# class label

pred1 = model_clt.labels_
pred1



