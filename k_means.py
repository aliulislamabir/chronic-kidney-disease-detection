# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans

# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import BernoulliNB

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Custom functions
def ClusterIndices(cluster_number, label):
    print (np.where(label == cluster_number)[0])
    
def ClusterValues(cluster_number, label):
    print (test_feature.iloc[np.where(label == cluster_number)[0],:].values)
    
def accuracy_score(matrix):
    tp_sum = 0
    score = 0
    np.array(matrix)
    for i in range(len(matrix)):
        tp_sum += matrix[i][i]
        
    score = (tp_sum / np.sum(matrix)) * 100
    return score

def Predict(vector):
    dist_list = []
    for i in range(len(centroids)):
        distance = [(a - b)**2 for a, b in zip(vector, centroids[i])]
        distance = math.sqrt(sum(distance))
        dist_list.append(distance)
        
    nos = len(np.where(labels == dist_list.index(min(dist_list)))[0]) # Number of samples

    print("Prediction Analysis:")
    print("==================================")
    print("Cluster number: " + str(dist_list.index(min(dist_list)) + 1))
    print("Cluster label: " + targets[dist_list.index(min(dist_list))])
    print("Number of samples: " + str(nos))
    print("----------------------------------")
    print("Summary:")
    print("Given observations seems to have larger amount of similarities with the samples of cluster " + 
          str(dist_list.index(min(dist_list)) + 1) + ". Preliminary diagnosis should be considered as suggested.")
    print("==================================")
"""    
def knn(train_data, test_data, train_label, test_label):
    for i in range(5,51,5):
        knn_e = KNeighborsClassifier(n_neighbors=i)
        knn_e.fit(train_data, train_label)
        pred_label = knn_e.predict(test_data)
        print("======================================================")
        print("\nReport using k=", i, ":\n")
        matrix = confusion_matrix(test_label, pred_label)
        print(matrix)
        print(classification_report(test_label, pred_label))
        accuracy = accuracy_score(matrix)
        print("\nAccuracy score: ", "%.2f" % accuracy, "%")
        print("======================================================")
"""        
dataframe = pd.read_csv("kidney_disease.csv")

# data encoding
encodings = {"yes":1, "good":1, "normal":1, "present":1,
             "no":0, "poor":0, "abnormal":0, "notpresent":0}
data = dataframe.replace(encodings).fillna(0)

# Spliting features and labels
feature_data = data.drop(["classification"], axis=1)
target_data = data.iloc[:,-1]

# Defining 'train data' and 'test data' # (if you need to devide the main dataset into train and test data)
train_feature, test_feature, train_target, test_target = train_test_split(feature_data, target_data, test_size = 0.3, random_state = 100)

# Finding names of unique targets
target_names = dataframe['classification'].unique().tolist()
test_targetc = pd.DataFrame(test_target, columns= ["Original labels"])

# Evaluation using k-Means
# KMeans clustering
kmeans = KMeans(n_clusters=len(target_names), random_state=100)
kmeans.fit(test_feature)
labels = kmeans.predict(test_feature) # K-means prediction
centroids = kmeans.cluster_centers_

# Fixing numeric labels into category
num_to_cat = {0:'ckd', 1:'notckd'}
labels_category = pd.DataFrame(labels,columns=["Predicted_labels"]).replace(num_to_cat)

# Reducing data dimensions using t-SNE
tsne = TSNE(n_components=2,perplexity=30,learning_rate=100,random_state=100)
embedded_data = tsne.fit_transform(test_feature)
dtp = pd.DataFrame(embedded_data, columns= ["embedded_feature_01", "embedded_feature_02"])
"""
# Printing cluster analytics
print("Cluster analytics:")
for cn in range(len(centroids)):
    print("==================================")
    print("Cluster " + str(cn+1) + " indices:")
    ClusterIndices(cn, custom_label) # Returns index of data points in a cluster
    print("----------------------------------")
    print("Cluster " + str(cn+1) + " values:")
    ClusterValues(cn, custom_label) # Returns datapoints of a cluster
    print("==================================")
"""    
# Preparing final dataframe of important columns to plot data
ploting_data = pd.DataFrame(pd.concat([dtp, labels_category, test_target], sort=False, ignore_index=True, axis=1).values,
                            columns=["embedded_feature_01", "embedded_feature_02", "Predicted_labels", "Original_labels"])

# Printing result analytics
print("Result analytics:")
c_matrix = confusion_matrix(test_target, labels_category)

print("==================================")
print("Confusion matrix:")
print(c_matrix)
print("----------------------------------")
print("Classification report:")
print(classification_report(test_target, labels_category))
print("Accuracy score: " + str("%.2f" % (accuracy_score(c_matrix))) + " %")
print("==================================")

# Ploting(scatter plot) raw datapoints according to original dataset
sns.lmplot(x="embedded_feature_01",y="embedded_feature_02",data=ploting_data,fit_reg=False)
# plt.savefig("Raw_datapoints.png")

# Ploting(scatter plot) original clusters according to original dataset
sns.lmplot(x="embedded_feature_01",y="embedded_feature_02",data=ploting_data,fit_reg=False,hue="Original_labels")
# plt.savefig("Original_datapoints.png")

# Ploting(scatter plot) kmeans predicted clusters
sns.lmplot(x="embedded_feature_01",y="embedded_feature_02",data=ploting_data,fit_reg=False,hue="Predicted_labels")
# plt.savefig("Predicted_datapoints.png")


