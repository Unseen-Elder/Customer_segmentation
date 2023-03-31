
# Customer Segmentation

This is a customer segmentation project that aims to investigate and interpret customer behavior for a wholesale distributor. The dataset includes the annual spending of 440 clients on different product categories. The goal of this project is to identify groups of customers with similar behavior, which can help improve business strategies and increase customer satisfaction.

To achieve this, we will follow a series of steps, starting with exploratory data analysis to understand the distribution of the data and identify any trends or patterns. We will then employ dimensionality reduction techniques, such as Principal Component Analysis (PCA) and Kernel Principal Component Analysis (KPCA), to transform the data into a more manageable form and identify the most important features that contribute to the variance in the data.

Next, we will use K-Means clustering algorithm to cluster the customers based on their behavior, and use the Elbow Method to determine the optimal number of clusters. Finally, we will visualize our clusters in an interactive way to analyze them and their differences more thoroughly.

By the end of this project, we aim to provide insights into customer behavior that can help improve business strategies and increase customer satisfaction. This project can also serve as a valuable learning experience for those interested in unsupervised learning, clustering algorithms, and visualization techniques.
## Dataset Overview
[The dataset](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers) set used in this project is the Wholesale Customers dataset, which contains information on the annual spending (in monetary units) of 440 customers across different product categories. The data set was donated by Margarida G. M. S. Cardoso and has since been widely used in machine learning projects related to clustering, classification, and regression.

### Dataset Characteristics

The Wholesale Customers dataset is a collection of data on the annual spending of various customers on different types of retail products. The data set comprises 440 instances, and each instance has the following eight attributes:

- Fresh: annual spending (in monetary units) on fresh products (Continuous)
- Milk: annual spending (in monetary units) on milk products (Continuous)
- Grocery: annual spending (in monetary units) on grocery products (Continuous)
- Frozen: annual spending (in monetary units) on frozen products (Continuous)
- Detergents_Paper: annual spending (in monetary units) on detergents and paper products (Continuous)
- Delicatessen: annual spending (in monetary units) on delicatessen products (Continuous)
- Channel: customers’ channel - Horeca (Hotel/Restaurant/Café) or Retail channel (Nominal)
- Region: customers’ region – Lisbon, Oporto or Other (Nominal)

The data set is multivariate, with each instance having eight attributes. The first six attributes are continuous numerical values representing the annual spending of a customer on different product categories. The last two attributes are categorical values representing the channel of customers (Horeca or Retail) and the region where the customers are located (Lisbon, Oporto, or Other).

## Project Outline

The project is divided into five tasks:

Task 1: Exploratory Data Analysis

Task 2: Principal Component Analysis

Task 3: Kernel Principal Component Analysis

Task 4: K-Means Clustering with Elbow Method

Task 5: Interactive Cluster Analysis

Each task is designed to achieve a specific objective in the project, and they are arranged in a logical order to ensure that the project is completed successfully. The details of each task will be discussed in subsequent sections.

### Task 1: Exploratory Data Analysis

In this task, I performed exploratory data analysis on the Wholesale Customers dataset. I loaded the data into a pandas dataframe and explored the data using various methods:

- `df.head()` to check the first few entries in the dataset.
- `df.info()` to get information about the data types and missing values.
- Converted the 'Channel' and 'Region' columns to categorical variables and then mapped them to their respective names.

```python
df = df.astype({'Channel' :'category','Region':'category'})
df['Channel'] = df['Channel'].map({1: 'HoReCa', 2: 'Retail'})
df['Region'] = df['Region'].map({1: 'Lisbon', 2: 'Porto', 3:'Other'})
```

I then created several visualizations to better understand the data:

- A bar plot of the number of customers in the HoReCa and Retail channels.
- A bar plot of the number of customers in each of the three regions (Lisbon, Porto, and Other).
- A histogram of the amount of money spent on each product category (Milk, Fresh, etc.) with 20 bins.
- A seaborn pairwise plot to explore the relationships between the different product categories. The first plot was created without any hue, the second plot had the 'Channel' column as the hue, and the third plot had the 'Region' column as the hue.

Overall, this exploratory data analysis gave us a better understanding of the dataset and the relationships between the different variables.

### Task 2: Principal Component Analysis (PCA)

In this task, I performed principal component analysis (PCA) on the given dataset to reduce the dimensionality of the data to two dimensions and visualize the data in two dimensions using a scatter plot.

First, I selected the relevant features from the dataset, converted them into a numpy array, and normalized the data using StandardScaler to ensure that all the features have a mean of zero and a standard deviation of one.

Then, I used the PCA algorithm from the scikit-learn library to transform the data into two principal components. I plotted the transformed data in two dimensions using a scatter plot, which showed the distribution of data points in the reduced feature space.

I also plotted the scatter plot for each feature against the two principal components to see if there is any correlation between the features and the principal components. Additionally, I plotted the scatter plot for each categorical variable (Channel and Region) against the two principal components to see if there is any clustering of data points based on those variables.

Overall, PCA helped me to identify patterns in the data and reduce the dimensionality of the dataset to make it easier to visualize and interpret.

### Task 3: Kernel PCA

In task 3, I performed dimensionality reduction using Kernel Principal Component Analysis (KPCA). KPCA is a nonlinear dimensionality reduction technique that allowed me to project high-dimensional data into a lower-dimensional space while preserving nonlinear relationships between the original features.

In this task, I used three different kernel functions:

- Polynomial kernel function with a degree of 2 (kernel='poly', degree=2)
- Radial basis function kernel (kernel='rbf')
- Cosine kernel (kernel='cosine')

I then applied KPCA to the standardized data and projected it onto a two-dimensional space, and visualized the results by plotting the transformed data against each pair of features and against the two categorical features Channel and Region.

Overall, KPCA provided a valuable tool for exploring the structure of the data and identifying nonlinear relationships between the features.


### Task 4 - K-Means Clustering with Elbow Method

In this task, I performed cluster analysis on the preprocessed data using KMeans algorithm from the Scikit-learn library. I used the KElbowVisualizer from the Yellowbrick library to determine the optimal number of clusters.

I chose to use the cosine kernel for this analysis because it provided the best visualisation results. I also used both silhouette score and distortion score to determine the optimal number of clusters. Based on the results, the optimal number of clusters was found to be 3, but I still chose to use 5 clusters because of market research and business reasons.

To visualize the clusters, I used scatter plots with different markers for each cluster, where each point in the plot represented a data point in the dataset. This allowed me to visually inspect the distribution of data points within each cluster for each feature and identify any patterns or insights.

Finally, I evaluated the clustering results by examining the data within each cluster to identify any patterns or insights that could be useful for further analysis. 

### Task 5 - Interactive Cluster Analysis

To visualize our results, we used a polar chart created with Plotly. We first normalized our data using the maximum value of each feature to ensure all values ranged between 0 and 1. Then we calculated the mean values of each feature within each cluster and plotted the resulting data as a polygon, with each vertex of the polygon representing a feature and each cluster represented by a different color.

The polygon for each cluster shows the average values of the features for that cluster, and the size of each polygon is proportional to the number of customers in that cluster. We also added text labels to each vertex of the polygon to display the actual values of the feature means.

Using this visualization, we can identify the key characteristics of each cluster and develop targeted marketing strategies accordingly.

You can access the chart by using this [Link](https://rawcdn.githack.com/Unseen-Elder/Customer_segmentation/ae0f191233b23d94e2fb810b1a34992d636d06a1/Result.html)

## Summary

This project is focused on customer segmentation for a wholesale distributor using unsupervised learning techniques. The dataset used is the Wholesale Customers dataset, which contains information on the annual spending of 440 customers across different product categories. The project is divided into five tasks, starting with exploratory data analysis to understand the distribution of the data and identify any trends or patterns. The subsequent tasks include dimensionality reduction techniques, such as Principal Component Analysis (PCA) and Kernel Principal Component Analysis (KPCA), K-Means clustering algorithm, and interactive cluster analysis. The project aims to identify groups of customers with similar behavior, which can help improve business strategies and increase customer satisfaction. The project can also serve as a valuable learning experience for those interested in unsupervised learning, clustering algorithms, and visualization techniques.
## Extra Resources

- [Principal Component Analysis (PCA)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Kernel Principal Component Analysis (KPCA)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)
- [K-Means Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Elbow Method](https://www.scikit-yb.org/en/latest/api/cluster/elbow.html)
