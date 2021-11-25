#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 01:58:49 2021

@author: riccelli
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns 

dataset = pd.read_csv("./Datasets/dataset_full.csv" )
logo_evaluation = dataset.iloc[0:9,:]
features = ['AGE',"Anger Frames >= Threshold","	Sadness Frames >= Threshold","	Disgust Frames >= Threshold", "Joy Frames >= Threshold","Surprise Frames >= Threshold","	Fear Frames >= Threshold", "Contempt Frames >= Threshold",'et','gsr']
x = logo_evaluation.loc[:, features].values
# Separating out the target
y = logo_evaluation.loc[:,['label']].values
# le = preprocessing.LabelEncoder()
# y = le.fit_transform(y)
# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
print (pca.explained_variance_)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, pd.DataFrame(y)], axis = 1)
finalDf.columns = ['principal component 1', 'principal component 2', 'label']

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['look', 'pinarello', 'trek','specialized']
colors = ['r', 'g', 'b','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(x)

df_subset = pd.DataFrame()
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
df_subset['label'] = y

plt.figure(figsize=(8,5))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    palette=sns.color_palette("dark", 4),
    data=df_subset,
    legend="full",
    sizes=(40, 40),
    alpha=1
)
    