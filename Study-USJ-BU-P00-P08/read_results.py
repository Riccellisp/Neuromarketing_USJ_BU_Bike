#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 20:56:49 2021

@author: riccelli
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
def ET_Results():
    
    d = pd.read_csv("./Eye Tracking/HL_INDIVIDUAL_FIXATION_TABLE.txt")
    
    cols  = d.columns
    
    participants = d["Respondent Name"].unique()
    
    stimulli = d["Stimulus Name"].unique()
    
    
    mean_fixation_start = []
    
    mean_duration = []
    
    subjects = []
    
    sts = []
    
    for stimu in stimulli:
        
        for subject in participants:
            
            is_stimu = d[d["Stimulus Name"] == stimu]
            is_sub =  is_stimu[is_stimu["Respondent Name"] == subject]
            mean_fixation_start.append(np.mean(is_sub["Fixation StartMs"]))
            mean_duration.append(np.mean(is_sub["Duration Ms"]))
            subjects.append(subject)
            sts.append(stimu)
            
    mean_fixation_start = np.array(mean_fixation_start)
    mean_duration = np.array(mean_duration)
    subjects = np.array(subjects)
    sts = np.array(sts)
    
    results = np.vstack([sts,subjects,mean_fixation_start,mean_duration]).T
    
    cols_results = ["stimulli","subject","mean_fixation_start","mean_duration"]
    
    results = pd.DataFrame(results, columns=cols_results)
    
    results.to_csv("./Eye Tracking/ET_results.csv")
    
def FC_results():
    
    emotions = ["Respondent Name","Label","Anger Frames >= Threshold","Sadness Frames >= Threshold",
                "Disgust Frames >= Threshold","Joy Frames >= Threshold",
                "Surprise Frames >= Threshold","Fear Frames >= Threshold",
                "Contempt Frames >= Threshold","Engagement Frames >= Threshold",
                "Attention Frames >= Threshold"]
    
    d = pd.read_csv("./Facial Coding/AFFDEX_Statistics.csv")
    d = d[emotions]
    cols  = d.columns
    
    participants = d["Respondent Name"].unique()
    
    stimulli = d["Label"].unique()
    
    subjects = []
    
    sts = []
    
    emotions_cols = []
    
    nonzero_emotions = []
    
    nonzero_columns = []
    
    count = 0

    # tick_labels = ['Disgust', 'Joy','Surprise', 'Engagement','Att.', 'Positive','Brow','Lip Corner', 'Smile','Inner','Eye Closure', 'Nose Wrinkle','Upper Lip', 'Lip Suck','Lip Press', 'Mouth Open','Lip Pucker', 'Cheek Raise','Dimpler', 'Eye Widen','Lip Stretch', 'Jaw Drop']
    
    for col in cols:
        if (col.find(">= Threshold") != -1):
            emotions_cols.append(col)
    
    results_cols = {x.replace('Frames >= Threshold', '').replace('', '') for x in emotions_cols}    

    for stimu in stimulli:
        is_stimu = d[d["Label"] == stimu]
        is_stimu = is_stimu[emotions_cols]
        perstimulli = is_stimu.sum()
        perstimulli.to_csv(f"./Facial Coding/perstimulli{stimu}.csv",header=False)
        
    for subject in participants:
        is_sub = d[d["Respondent Name"] == subject]
        is_sub.to_csv(f"./Facial Coding/persubject{subject}.csv",header=emotions,index=False)
        
def GSR_results():
    d = pd.read_csv("./GSR/GSR Summary Scores.csv")
    cols  = d.columns
    
    peaks = ["Respondent Name","Label","Has Peaks","Peak Count"]
    
    d = d[peaks]
    
    peaks_metrics = ["Has Peaks","Peak Count"]
    
    participants = d["Respondent Name"].unique()
    stimulli = d["Label"].unique()
    ################################## per stimulus############################
    for stimu in stimulli:
        is_stimu = d[d["Label"] == stimu]
        perstimulli = is_stimu[peaks_metrics].sum()
        perstimulli.to_csv(f"./GSR/perstimulli{stimu}_scores.csv",header=False)
    ################################# per subject #############################
    #for stimu in stimulli:
        #is_stimu = d[d["Label"] == stimu]
    for subject in participants:
        is_sub = d[d["Respondent Name"] == subject]
        is_sub.to_csv(f"./GSR/persubject{subject}_scores.csv",header=False,index=False)
        
def Survey_results():
    # Need to adjust survey csv
    
    d = pd.read_csv("./Survey/MERGED_SURVEY_RESPONSE_MATRIX.csv")
    cols = d.columns
    results_cols = ["RESPONDENT","AGE","LABELID_Survey-FavoriteBrand_Q1-Favorite-Brand","LABELID_Survey-FavoriteProduct_Q1-Favorite-Brand","LABELID_Survey-FavoriteVideoAd_Q1-Favorite-Brand"]
    results = d[results_cols]
    results.to_csv(f"./Survey/filtered-results.csv",header=True,index=False)
    
def ExtractFromFavoriteLogoPerParticipant (favorite_participants,stimulli,logoname,experiment):
    favorite_participants = favorite_participants["RESPONDENT"].values
    results_gsr = []
    results_et = []
    results_fc = []
    for participant in favorite_participants:
        gsr= pd.read_csv(f"./GSR/persubject{participant}_scores.csv",names=['RESPONDENT','STIMULI','HASPkEAKS','PEAKSCOUNT'])
        et = pd.read_csv(f"./Eye Tracking/ET_results.csv",names=['STIMULI','RESPONDENT','MEAN-FIXATION-START','MEAN-DURATION'])
        fc = pd.read_csv(f"./Facial Coding/persubject{participant}.csv",skiprows=(1),names=['RESPONDENT','STIMULI','Anger Frames >= Threshold','	Sadness Frames >= Threshold','	Disgust Frames >= Threshold',	'Joy Frames >= Threshold',	'Surprise Frames >= Threshold','	Fear Frames >= Threshold',	'Contempt Frames >= Threshold','	Engagement Frames >= Threshold','	Attention Frames >= Threshold'])
        gsr = gsr[gsr["STIMULI"]==stimulli]
        gsrpeaks = gsr['PEAKSCOUNT'].values
        results_gsr.append(gsrpeaks)
        et = et[et["STIMULI"]==stimulli]
        et = et[et["RESPONDENT"]==participant]
        # breakpoint()
        mean_duration = et["MEAN-DURATION"].values
        results_et.append(mean_duration)
        fc = fc[fc["STIMULI"]==stimulli]
        fc = fc[fc["RESPONDENT"]==participant]
        results_fc.append(fc)
        if not os.path.exists(f'./Facial Coding/Results_per_participant_per_{experiment}_{logoname}'):
            os.makedirs(f'./Facial Coding/Results_per_participant_per_{experiment}_{logoname}')
        fc.to_csv(f"./Facial Coding/Results_per_participant_per_{experiment}_{logoname}/{participant}logo-{logoname}.csv",index=False)
        
    return results_gsr,results_et,results_fc    


def ExtractFromFavoriteLogoPerParticipant2 (favorite_participants,stimulli,logoname):
    favorite_participants = favorite_participants["RESPONDENT"].values
    results_gsr = []
    results_et = []
    results_fc = []
    for participant in favorite_participants:
        gsr= pd.read_csv(f"./GSR/persubject{participant}_scores.csv",names=['RESPONDENT','STIMULI','HASPkEAKS','PEAKSCOUNT'])
        et = pd.read_csv(f"./Eye Tracking/ET_results.csv",names=['STIMULI','RESPONDENT','MEAN-FIXATION-START','MEAN-DURATION'])
        fc = pd.read_csv(f"./Facial Coding/persubject{participant}.csv",skiprows=(1),names=['RESPONDENT','STIMULI','Anger Frames >= Threshold','	Sadness Frames >= Threshold','	Disgust Frames >= Threshold',	'Joy Frames >= Threshold',	'Surprise Frames >= Threshold','	Fear Frames >= Threshold',	'Contempt Frames >= Threshold','	Engagement Frames >= Threshold','	Attention Frames >= Threshold'])
        gsr = gsr[gsr["STIMULI"]==stimulli]
        gsrpeaks = gsr['PEAKSCOUNT'].values
        results_gsr.append(gsrpeaks)
        et = et[et["STIMULI"]==stimulli]
        et = et[et["RESPONDENT"]==participant]
        # breakpoint()
        mean_duration = et["MEAN-DURATION"].values
        results_et.append(mean_duration)
        fc = fc[fc["STIMULI"]==stimulli]
        fc = fc[fc["RESPONDENT"]==participant]
        results_fc.append(fc)
        if not os.path.exists(f'./Facial Coding/Results_per_participant_per_logo_{logoname}'):
            os.makedirs(f'./Facial Coding/Results_per_participant_per_logo_{logoname}')
        fc.to_csv(f"./Facial Coding/Results_per_participant_per_logo_{logoname}/{participant}logo-{logoname}.csv",index=False)
        
    return results_gsr,results_et,results_fc

def CreateDataset (results_gsr_favorite,results_et_favorite,results_fc_favorite,ages,logoname,experiment):
    if  results_fc_favorite:
        dataset = pd.concat(results_fc_favorite)
        dataset['gsr'] = np.array(results_gsr_favorite)
        if experiment != 'video':
            dataset['et'] = np.array(results_et_favorite)
        if logoname =='treks':
            
            breakpoint()
        dataset.insert(1, 'AGE', ages)
        dataset['experiment'] = experiment
        dataset['label'] = logoname
        dataset.reset_index(inplace=True)
        if not os.path.exists('./Datasets'):
            os.makedirs('./Datasets')
        dataset.to_csv(f"./Datasets/dataset_brand_{logoname}_favorite_{experiment}.csv",index=False)
        return dataset
    else:
        print(f"Empty dataframe, there is no participant that liked this {logoname} in {experiment} evaluation?")
        return None
    
def MergeDatasets():
    print('a')
    
def Dataset():
    d = pd.read_csv("./Survey/filtered-results.csv")
    cols = d.columns
    
    favorite_logo_cols  = ["RESPONDENT","AGE","LABELID_Survey-FavoriteBrand_Q1-Favorite-Brand"]
    favorite_logo =  d[favorite_logo_cols]
    favorite_logo.reset_index(inplace=True)
    favorite_logo.drop(columns='index',inplace=True)
    
    favorite_logo_look = favorite_logo[favorite_logo["LABELID_Survey-FavoriteBrand_Q1-Favorite-Brand"] == "Look"]
    favorite_logo_pinarello = favorite_logo[favorite_logo["LABELID_Survey-FavoriteBrand_Q1-Favorite-Brand"] == "Pinarello"]
    favorite_logo_trek = favorite_logo[favorite_logo["LABELID_Survey-FavoriteBrand_Q1-Favorite-Brand"] == "Trek"]
    favorite_logo_specialized = favorite_logo[favorite_logo["LABELID_Survey-FavoriteBrand_Q1-Favorite-Brand"] == "Specialized"]
    
    favorite_product_cols  = ["RESPONDENT","AGE","LABELID_Survey-FavoriteProduct_Q1-Favorite-Brand"]
    favorite_product = d[favorite_product_cols]
    favorite_product.reset_index(inplace=True)
    favorite_product.drop(columns='index',inplace=True)
    
    favorite_product_look = favorite_product[favorite_product["LABELID_Survey-FavoriteProduct_Q1-Favorite-Brand"] == "Look"]
    favorite_product_pinarello = favorite_product[favorite_product["LABELID_Survey-FavoriteProduct_Q1-Favorite-Brand"] == "Pinarello"]
    favorite_product_trek = favorite_product[favorite_product["LABELID_Survey-FavoriteProduct_Q1-Favorite-Brand"] == "Trek"]
    favorite_product_specialized = favorite_product[favorite_product["LABELID_Survey-FavoriteProduct_Q1-Favorite-Brand"] == "Specialized"]
    
    
    favorite_video_cols  = ["RESPONDENT","AGE",    "LABELID_Survey-FavoriteVideoAd_Q1-Favorite-Brand"]
    favorite_video = d[favorite_video_cols]
    favorite_video.reset_index(inplace=True)
    favorite_video.drop(columns='index',inplace=True)
    
    favorite_video_look = favorite_video[favorite_video["LABELID_Survey-FavoriteVideoAd_Q1-Favorite-Brand"] == "Look"]
    favorite_video_pinarello = favorite_video[favorite_video["LABELID_Survey-FavoriteVideoAd_Q1-Favorite-Brand"] == "Pinarello"]
    favorite_video_trek = favorite_video[favorite_video["LABELID_Survey-FavoriteVideoAd_Q1-Favorite-Brand"] == "Trek"]
    favorite_video_specialized = favorite_video[favorite_video["LABELID_Survey-FavoriteVideoAd_Q1-Favorite-Brand"] == "Specialized"]
    

    # logo01
    # logo02_name-white
    # logo03_name-white
    # logo04_name-white
    
    # logo-product01
    # logo-product02
    # logo-product03
    # logo-product04

    # video01 - LOOK
    # video02 - Pinarello
    # video03 - Trek
    # video04 - Specialized


    results_gsr_favorite_logo_look,results_et_favorite_logo_look,results_fc_favorite_logo_look = ExtractFromFavoriteLogoPerParticipant (favorite_logo_look,'logo01','look','logo')
    dataset_logo_look = CreateDataset(results_gsr_favorite_logo_look,results_et_favorite_logo_look,results_fc_favorite_logo_look,favorite_logo_look['AGE'].values,'look','logo')
    
    results_gsr_favorite_logo_pinarello,results_et_favorite_logo_pinarello,results_fc_favorite_logo_pinarello = ExtractFromFavoriteLogoPerParticipant (favorite_logo_pinarello,'logo02_name-white','pinarello','logo')
    dataset_logo_pinarello = CreateDataset(results_gsr_favorite_logo_pinarello,results_et_favorite_logo_pinarello,results_fc_favorite_logo_pinarello,favorite_logo_pinarello['AGE'].values,'pinarello','logo')

    results_gsr_favorite_logo_trek,results_et_favorite_logo_trek,results_fc_favorite_logo_trek = ExtractFromFavoriteLogoPerParticipant (favorite_logo_trek,'logo03_name-white','trek','logo')
    dataset_logo_trek = CreateDataset(results_gsr_favorite_logo_trek,results_et_favorite_logo_trek,results_fc_favorite_logo_trek,favorite_logo_trek['AGE'].values,'trek','logo')

    results_gsr_favorite_logo_specialized,results_et_favorite_logo_specialized,results_fc_favorite_logo_specialized = ExtractFromFavoriteLogoPerParticipant (favorite_logo_specialized,'logo04_name-white','specialized','logo')
    dataset_logo_specialized = CreateDataset(results_gsr_favorite_logo_specialized,results_et_favorite_logo_specialized,results_fc_favorite_logo_specialized,favorite_logo_specialized['AGE'].values,'specialized','logo')
    
    
    
    ############################################## Product #####################################



    results_gsr_favorite_product_look,results_et_favorite_product_look,results_fc_favorite_product_look = ExtractFromFavoriteLogoPerParticipant (favorite_product_look,'logo-product01','look','product')
    dataset_product_look = CreateDataset(results_gsr_favorite_product_look,results_et_favorite_product_look,results_fc_favorite_product_look,favorite_product_look['AGE'].values,'look','product')

    results_gsr_favorite_product_pinarello,results_et_favorite_product_pinarello,results_fc_favorite_product_pinarello = ExtractFromFavoriteLogoPerParticipant (favorite_product_pinarello,'logo-product02','pinarello','product')
    dataset_product_pinarello = CreateDataset(results_gsr_favorite_product_pinarello,results_et_favorite_product_pinarello,results_fc_favorite_product_pinarello,favorite_product_pinarello['AGE'].values,'pinarello','product')

    results_gsr_favorite_product_trek,results_et_favorite_product_trek,results_fc_favorite_product_trek = ExtractFromFavoriteLogoPerParticipant (favorite_product_trek,'logo-product03','trek','product')
    dataset_product_trek = CreateDataset(results_gsr_favorite_product_trek,results_et_favorite_product_trek,results_fc_favorite_product_trek,favorite_product_trek['AGE'].values,'trek','product')

    results_gsr_favorite_product_specialized,results_et_favorite_product_specialized,results_fc_favorite_product_specialized  = ExtractFromFavoriteLogoPerParticipant (favorite_product_specialized,'logo-product04','specialized','product')
    dataset_product_specialized = CreateDataset(results_gsr_favorite_product_specialized,results_et_favorite_product_specialized,results_fc_favorite_product_specialized,favorite_product_specialized['AGE'].values,'specialized','product')

    
    
    ########################################## Video #########################################


    results_gsr_favorite_video_look,results_et_favorite_video_look,results_fc_favorite_video_look = ExtractFromFavoriteLogoPerParticipant (favorite_video_look,'video01 - LOOK','look','video')
    dataset_video_look = CreateDataset(results_gsr_favorite_video_look,results_et_favorite_video_look,results_fc_favorite_video_look,favorite_video_look['AGE'].values,'look','video')

    results_gsr_favorite_video_pinarello,results_et_favorite_video_pinarello,results_fc_favorite_video_pinarello  = ExtractFromFavoriteLogoPerParticipant (favorite_video_pinarello,'video02 - Pinarello','pinarello','video')
    dataset_video_pinarello = CreateDataset(results_gsr_favorite_video_pinarello,results_et_favorite_video_pinarello,results_fc_favorite_video_pinarello,favorite_video_pinarello['AGE'].values,'pinarello','video')

    results_gsr_favorite_video_trek,results_et_favorite_video_trek,results_fc_favorite_video_trek = ExtractFromFavoriteLogoPerParticipant (favorite_video_trek,'video03 - Trek','trek','video')
    dataset_video_trek = CreateDataset(results_gsr_favorite_video_trek,results_et_favorite_video_trek,results_fc_favorite_video_trek,favorite_video_trek['AGE'].values,'trek','video')

    results_gsr_favorite_video_specialized,results_et_favorite_video_specialized,results_fc_favorite_video_specialized  = ExtractFromFavoriteLogoPerParticipant (favorite_video_specialized,'video04 - Specialized','specialized','video')
    dataset_video_specialized = CreateDataset(results_gsr_favorite_video_specialized,results_et_favorite_video_specialized,results_fc_favorite_video_specialized,favorite_video_specialized['AGE'].values,'specialized','video')

    
    dataset = pd.concat([dataset_logo_look,dataset_logo_pinarello,dataset_logo_trek,dataset_logo_specialized,dataset_product_look,
                         dataset_product_pinarello,dataset_product_trek,dataset_product_specialized,dataset_video_look,
                         dataset_video_pinarello,dataset_video_trek,dataset_video_specialized],ignore_index=True)
    
    dataset.drop(columns='index',inplace=True)
    

    dataset.to_csv(f"./Datasets/dataset_full.csv",index=False)
    
    # adults = favorite_logo[]
    # youngs = 
    return dataset
    
    

def main():
    # ET_Results()
    # FC_results()
    # GSR_results()
    # Survey_results()
    dataset = Dataset()
if __name__ == "__main__":
    main()
