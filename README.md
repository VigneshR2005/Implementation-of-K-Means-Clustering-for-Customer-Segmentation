# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
    -Choose the number of clusters (K): 
          Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

    -Initialize cluster centroids: 
        Randomly select K data points from your dataset as the initial centroids of the clusters.

    -Assign data points to clusters: 
      Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically  done using Euclidean distance, but other distance metrics can also be used.

    -Update cluster centroids: 
      Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

    -Repeat steps 3 and 4: 
      Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

    -Evaluate the clustering results: 
      Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

    -Select the best clustering solution: 
      If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements
  ```

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: R VIGNESH
RegisterNumber:  212222230172
*/
```
```py
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("/content/Dataset-20230524.zip")
data

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])

y_pred = km.predict(data.iloc[:,3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="yellow",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="pink",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="purple",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:
### DATA.HEAD():
![8=1](https://github.com/Senthamil1412/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119120228/e40574cb-ed1e-4086-974b-ef61e4c5e298)


### DATA.info():
![8=2](https://github.com/Senthamil1412/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119120228/e5224707-b3ac-4a4d-96b4-a66008d4229c)


### NULL VALUES:
![8=3](https://github.com/Senthamil1412/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119120228/6bf11175-f063-4e8d-aeb3-c8348b0e923a)

### ELBOW GRAPH:
![8=4](https://github.com/Senthamil1412/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119120228/c0254078-51b1-4516-a9b4-fbe0ec18567b)

### CLUSTER FORMATION:
![8=5](https://github.com/Senthamil1412/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119120228/269b52fe-bf88-4775-816c-5d02d66c8082)

### PREDICICTED VALUE:
![8=6](https://github.com/Senthamil1412/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119120228/737b5085-fecb-487e-b291-220de5f4665a)

### FINAL GRAPH(D/O):
![8=7](https://github.com/Senthamil1412/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119120228/7b3a8093-1d2b-4557-9555-0149a411dc31)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
