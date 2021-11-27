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
from sklearn.ensemble import RandomForestClassifier
import time
import os


def Feature_selection(dataset,experiment,classes):
    
    if not os.path.exists(f'./FeatureSelection/{experiment}'):
        os.makedirs(f'./FeatureSelection/{experiment}')
    
    if experiment == 'video':
        features = ["Anger Frames >= Threshold","	Sadness Frames >= Threshold","	Disgust Frames >= Threshold", "Joy Frames >= Threshold","Surprise Frames >= Threshold","	Fear Frames >= Threshold", "Contempt Frames >= Threshold",'gsr']
        # features_names = ['AGE',"Anger","Sadness","Disgust", "Joy","Surprise","	Fear", "Contempt",'et','gsr']
        features_names = ["Anger","Sadness","Disgust", "Joy","Surprise","Fear", "Contempt",'gsr']
    else:
        # features = ['AGE',"Anger Frames >= Threshold","	Sadness Frames >= Threshold","	Disgust Frames >= Threshold", "Joy Frames >= Threshold","Surprise Frames >= Threshold","	Fear Frames >= Threshold", "Contempt Frames >= Threshold",'et','gsr']
        features = ["Anger Frames >= Threshold","	Sadness Frames >= Threshold","	Disgust Frames >= Threshold", "Joy Frames >= Threshold","Surprise Frames >= Threshold","	Fear Frames >= Threshold", "Contempt Frames >= Threshold",'et','gsr']
        # features_names = ['AGE',"Anger","Sadness","Disgust", "Joy","Surprise","	Fear", "Contempt",'et','gsr']
        features_names = ["Anger","Sadness","Disgust", "Joy","Surprise","Fear", "Contempt",'et','gsr']
        
    x = dataset.loc[:, features].values
    # Separating out the target
    y = dataset.loc[:,['label']].values
    # le = preprocessing.LabelEncoder()
    # y = le.fit_transform(y)
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    
    ################################### Feature importance ########################
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
    plt.savefig(f'./FeatureSelection/{experiment}/pcaplot_variance_{pca.explained_variance_ratio_.cumsum()}.png')
    
    
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
        palette=sns.color_palette("dark", classes),
        data=df_subset,
        legend="full",
        sizes=(40, 40),
        alpha=1
    )
    plt.savefig(f'./FeatureSelection/{experiment}/tsneplot.png')
    ################################### Feature importance ########################
    forest = RandomForestClassifier(random_state=0)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    forest.fit(x, y)
    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time
    
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    
    forest_importances = pd.Series(importances, index=features_names)
    
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()    
    plt.savefig(f'./FeatureSelection/{experiment}/feature_importance.png')
    
def main():
    dataset = pd.read_csv("./Datasets/dataset_full.csv" )
    '''
        LOGO
    '''
    # dataset = dataset.iloc[0:9,:]
    # experiment = 'logo'
    # n_classes = len(dataset['label'].value_counts())
    # Feature_selection(dataset,experiment,n_classes)
    '''
        PRODUCT
    '''
    # dataset = dataset.iloc[9:18,:]
    # experiment = 'product'
    # n_classes = len(dataset['label'].value_counts())
    # Feature_selection(dataset,experiment,n_classes)
    '''
        VIDEO
    '''
    # dataset = dataset.iloc[18:27,:]
    # n_classes = len(dataset['label'].value_counts())
    # experiment = 'video'
    # Feature_selection(dataset,experiment,n_classes)
if __name__ == "__main__":
    main()
