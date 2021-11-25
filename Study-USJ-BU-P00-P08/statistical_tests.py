#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#     ["Anger Frames >= Threshold",
# 	"Sadness Frames >= Threshold",
# 	"Disgust Frames >= Threshold",
#     "Joy Frames >= Threshold",
#     "Surprise Frames >= Threshold",
# 	"Fear Frames >= Threshold",
#     "Contempt Frames >= Threshold"]
    
"""
Created on Tue Nov 23 01:41:44 2021

@author: riccelli
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

dataset = pd.read_csv("./Datasets/dataset_full.csv" )

def Prepare_metrics (dataset,logoname):
    
    dataset = dataset[dataset["label"] == logoname]
    
    # dataset_emotions = dataset.drop(['RESPONDENT','AGE','STIMULI',"Anger Frames >= Threshold","	Sadness Frames >= Threshold","	Disgust Frames >= Threshold", "Joy Frames >= Threshold","Surprise Frames >= Threshold","	Fear Frames >= Threshold", "Contempt Frames >= Threshold",'et','experiment','label'],axis=1)
    dataset_emotions = dataset.drop(['RESPONDENT','AGE','STIMULI','et','experiment','label'],axis=1)

    brand_phase_evaluation_cumulated = dataset_emotions.sum()
    mean_et = np.mean(dataset['et'])
    brand_phase_evaluation_cumulated['et_mean'] = mean_et
    return brand_phase_evaluation_cumulated

################################### LOGO ################################################

logo_evaluation = dataset.iloc[0:9,:]

look = Prepare_metrics(logo_evaluation,'look')
pinarello = Prepare_metrics(logo_evaluation,'pinarello')
trek = Prepare_metrics(logo_evaluation,'trek')
specialized = Prepare_metrics(logo_evaluation,'specialized')

stat_look_pinarelo, p_look_pinarelo =  mannwhitneyu(look.values, pinarello.values)
stat_look_trek,p_look_trek = mannwhitneyu(look.values, trek.values)
stat_look_specialized,p_look_specialized = mannwhitneyu(look.values, specialized.values)

stat_pinarelo_trek, p_pinerello_trek =  mannwhitneyu(pinarello.values, trek.values)
stat_look_trek,p_look_trek = mannwhitneyu(look.values, trek.values)
stat_look_specialized,p_look_specialized = mannwhitneyu(look.values, specialized.values)
# look_logo_evaluation = logo_evaluation[logo_evaluation["label"] == "look"]
# look_logo_evaluation_emotions = look_logo_evaluation.drop(['RESPONDENT','AGE','STIMULI','et','experiment','label'],axis=1)
# look_logo_evaluation_cumulated = look_logo_evaluation_emotions.sum()
# mean_et = np.mean(look_logo_evaluation['et'])
# look_logo_evaluation_cumulated['et_mean'] = mean_et

# pinarello_logo_evaluation = logo_evaluation[logo_evaluation["label"] == "pinarello"]
# pinarello_logo_evaluation.reset_index(inplace=True)
# pinarello_logo_evaluation.drop(columns='index',inplace=True)
# pinarello_logo_evaluation_emotions = pinarello_logo_evaluation.drop(['RESPONDENT','AGE','STIMULI','et','experiment','label'],axis=1)
# pinarello_logo_evaluation_cumulated = pinarello_logo_evaluation_emotions.sum()
# mean_et = np.mean(pinarello_logo_evaluation['et'])
# pinarello_logo_evaluation_cumulated['et_mean'] = mean_et


# trek_logo_evaluation = logo_evaluation[logo_evaluation["label"] == "trek"]
# trek_logo_evaluation.reset_index(inplace=True)
# trek_logo_evaluation.drop(columns='index',inplace=True)
# trek_logo_evaluation_emotions = trek_logo_evaluation.drop(['RESPONDENT','AGE','STIMULI','et','experiment','label'],axis=1)
# trek_logo_evaluation_cumulated = trek_logo_evaluation_emotions.sum()
# mean_et = np.mean(trek_logo_evaluation['et'])
# trek_logo_evaluation_cumulated['et_mean'] = mean_et


# specialized_logo_evaluation = logo_evaluation[logo_evaluation["label"] == "specialized"]
# specialized_logo_evaluation.reset_index(inplace=True)
# specialized_logo_evaluation.drop(columns='index',inplace=True)
# specialized_logo_evaluation_emotions = specialized_logo_evaluation.drop(['RESPONDENT','AGE','STIMULI','et','experiment','label'],axis=1)
# specialized_logo_evaluation_cumulated = specialized_logo_evaluation_emotions.sum()
# mean_et = np.mean(specialized_logo_evaluation['et'])
# specialized_logo_evaluation_cumulated['et_mean'] = mean_et

# stat_pinarello_look_logo, p_pinarello_look_logo = mannwhitneyu(pinarello_logo_evaluation_cumulated.values, look_logo_evaluation_cumulated.values)
# stat_look_pinarello_logo, p_pinarello_look_logo = mannwhitneyu(look_logo_evaluation_cumulated.values,pinarello_logo_evaluation_cumulated.values)
# stat_look_trek_logo, p_look_trek_logo = mannwhitneyu( look_logo_evaluation_cumulated.values,trek_logo_evaluation_cumulated)
# stat_look_specialized_logo, look_specialized_logo = mannwhitneyu(look_logo_evaluation_cumulated.values,specialized_logo_evaluation_cumulated.values)


# stat, p = mannwhitneyu(pinarello_logo_evaluation_cumulated.values, look_logo_evaluation_cumulated.values)
# stat, p = mannwhitneyu(pinarello_logo_evaluation_cumulated.values, look_logo_evaluation_cumulated.values)
# stat, p = mannwhitneyu(pinarello_logo_evaluation_cumulated.values, look_logo_evaluation_cumulated.values)


################################### PRODUCT ################################################

product_evaluation = dataset.iloc[9:18,:]
look = Prepare_metrics(product_evaluation,'look')
pinarello = Prepare_metrics(product_evaluation,'pinarello')
trek = Prepare_metrics(product_evaluation,'trek')
specialized = Prepare_metrics(product_evaluation,'specialized')

stat_look_pinarelo, p_look_pinarelo =  mannwhitneyu(look.values, pinarello.values)
stat_look_trek,p_look_trek = mannwhitneyu(look.values, trek.values)
stat_look_specialized,p_look_specialized = mannwhitneyu(look.values, specialized.values)

breakpoint()










look_product_evaluation = product_evaluation[product_evaluation["label"] == "look"]
look_product_evaluation_emotions = look_product_evaluation.drop(['RESPONDENT','AGE','STIMULI','et','experiment','label'],axis=1)
look_product_evaluation_cumulated = look_product_evaluation_emotions.sum()
mean_et = np.mean(look_product_evaluation['et'])
look_product_evaluation_cumulated['et_mean'] = mean_et

pinarello_product_evaluation = product_evaluation[product_evaluation["label"] == "pinarello"]
pinarello_product_evaluation.reset_index(inplace=True)
pinarello_product_evaluation.drop(columns='index',inplace=True)
pinarello_product_evaluation_emotions = pinarello_product_evaluation.drop(['RESPONDENT','AGE','STIMULI','et','experiment','label'],axis=1)
pinarello_product_evaluation_cumulated = pinarello_product_evaluation_emotions.sum()
mean_et = np.mean(pinarello_product_evaluation['et'])
pinarello_product_evaluation_cumulated['et_mean'] = mean_et

trek_product_evaluation = product_evaluation[product_evaluation["label"] == "trek"]
trek_product_evaluation.reset_index(inplace=True)
trek_product_evaluation.drop(columns='index',inplace=True)
trek_product_evaluation_emotions = trek_product_evaluation.drop(['RESPONDENT','AGE','STIMULI','et','experiment','label'],axis=1)
trek_product_evaluation_cumulated = trek_product_evaluation_emotions.sum()
mean_et = np.mean(trek_product_evaluation['et'])
trek_product_evaluation_cumulated['et_mean'] = mean_et

specialized_product_evaluation = product_evaluation[product_evaluation["label"] == "specialized"]
specialized_product_evaluation.reset_index(inplace=True)
specialized_product_evaluation.drop(columns='index',inplace=True)
specialized_product_evaluation_emotions = specialized_product_evaluation.drop(['RESPONDENT','AGE','STIMULI','et','experiment','label'],axis=1)
specialized_product_evaluation_cumulated = specialized_product_evaluation_emotions.sum()
mean_et = np.mean(specialized_product_evaluation['et'])
specialized_product_evaluation_cumulated['et_mean'] = mean_et

################################### VIDEO ################################################
video_evaluation = dataset.iloc[18:27,:]
look_video_evaluation = video_evaluation[video_evaluation["label"] == "look"]
look_video_evaluation_emotions = look_video_evaluation.drop(['RESPONDENT','AGE','STIMULI','et','experiment','label'],axis=1)
look_video_evaluation_cumulated = look_video_evaluation_emotions.sum()
mean_et = np.mean(look_video_evaluation['et'])
look_video_evaluation_cumulated['et_mean'] = mean_et

pinarello_video_evaluation = video_evaluation[video_evaluation["label"] == "pinarello"]
pinarello_video_evaluation.reset_index(inplace=True)
pinarello_video_evaluation.drop(columns='index',inplace=True)
pinarello_video_evaluation_emotions = pinarello_video_evaluation.drop(['RESPONDENT','AGE','STIMULI','et','experiment','label'],axis=1)
pinarello_video_evaluation_cumulated = pinarello_video_evaluation_emotions.sum()
mean_et = np.mean(pinarello_video_evaluation['et'])
pinarello_video_evaluation_cumulated['et_mean'] = mean_et

trek_video_evaluation = video_evaluation[video_evaluation["label"] == "trek"]
trek_video_evaluation.reset_index(inplace=True)
trek_video_evaluation.drop(columns='index',inplace=True)
trek_video_evaluation_emotions = trek_video_evaluation.drop(['RESPONDENT','AGE','STIMULI','et','experiment','label'],axis=1)
trek_video_evaluation_cumulated = trek_video_evaluation_emotions.sum()
mean_et = np.mean(trek_video_evaluation['et'])
trek_video_evaluation_cumulated['et_mean'] = mean_et

specialized_video_evaluation = video_evaluation[video_evaluation["label"] == "specialized"]
specialized_video_evaluation.reset_index(inplace=True)
specialized_video_evaluation.drop(columns='index',inplace=True)
specialized_video_evaluation_emotions = specialized_video_evaluation.drop(['RESPONDENT','AGE','STIMULI','et','experiment','label'],axis=1)
specialized_video_evaluation_cumulated = specialized_video_evaluation_emotions.sum()
mean_et = np.mean(specialized_video_evaluation['et'])
specialized_video_evaluation_cumulated['et_mean'] = mean_et