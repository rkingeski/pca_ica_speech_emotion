# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 18:04:42 2024

@author: rking
"""


import numpy as np
import pandas as pd
import opensmile
from scipy.stats import anderson, kruskal
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, FastICA
from sklearn import preprocessing
import glob
  

    
wavs =[]


##################################################################################
#!!
#IMPORTANT! CHANGE THE PATH FOR THE DATABASE!!
#!!
##################################################################################

base_path = r"G:\Meu Drive\features_py\RAVDESS\**\*.wav*"
for file in glob.glob(base_path, recursive=True):
    print(f"WAV File: {file}")
    wavs.append(file)
 
    
'''

Filename identifiers 

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

'''
    
 
    
emotions = []

for i in range(len(wavs)):
    if wavs[i][-18:-16] == '01':
        emotions.append('neutral')
    elif wavs[i][-18:-16] == '02' :
        emotions.append('calm')
    elif wavs[i][-18:-16] == '03' :
        emotions.append('happiness')
    elif wavs[i][-18:-16] == '04':
        emotions.append('sadness')
    elif wavs[i][-18:-16] == '05':
        emotions.append('anger')
    elif wavs[i][-18:-16] == '06' :
        emotions.append('fear')
    elif wavs[i][-18:-16] == '07' :
        emotions.append('disgust')
    elif wavs[i][-18:-16] == '08' :
        emotions.append('suprise')
        
        
        
#db = audb.load('emodb') #load database

#df = db.tables['emotion'].df #load table with emotions


#OpenSmile package to extract audio features, select the configurations and the group of features
smile = opensmile.Smile(
   #feature_set=opensmile.FeatureSet.eGeMAPSv02,
   feature_set=opensmile.FeatureSet.ComParE_2016,
   feature_level=opensmile.FeatureLevel.Functionals,
)


feats_df = smile.process_files(wavs) #load the features

d = {'file': wavs, 'emotion': emotions}
df = pd.DataFrame(d)

#normalize


feats_dfn = pd.DataFrame(preprocessing.normalize(feats_df, axis=0), columns = feats_df.columns)




#feat_columns = feats_df.columns

#filter data using variance, this will remove the constant and quasi constant features
def filter_quasi_constants(feats_df, threshold=0.00001):

    #variance for each feature
    #variances = feats_df.var()

    #VarianceThreshold object
    selector = VarianceThreshold(threshold=threshold)

    #fit the selector to the data
    selector.fit(feats_df)

    #select the indices of features to keep
    indices_to_keep = selector.get_support(indices=True)

    #slected features by indices
    selected_features = feats_df.iloc[:, indices_to_keep]

    return selected_features

ffeats_df = filter_quasi_constants(feats_dfn)


#now for ICA we can use minmaxscaler or standardzation and normalize or just standardzation

#standardzation mean 0 std 1
scaler = preprocessing.StandardScaler().fit(ffeats_df)

#minmax range -1 to 1
#scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))


feats_dfs = pd.DataFrame(scaler.fit_transform(ffeats_df),columns=ffeats_df.columns)

##function to test the normality

def distribuition(data, confiability):

    if (confiability == 85):
        conf = 0
    if (confiability == 90):
        conf = 1
    if (confiability == 95):
        conf = 2
    if (confiability == 97.2):
        conf = 3
    if (confiability == 99):
        conf = 4
        
        
    stats = []
    cvalue = []
    i=0
    
    for columns in data:
        res = anderson(data.iloc[:,i])
        stats.append(res.statistic)
        cvalue.append(res.critical_values[conf])
        i=i+1
    
    norm_index = []
    norm_name = []
    notnorm_index = []
    notnorm_name = []
    k=0
    j=0    
    
    for i in range(len(cvalue)):
        if stats[i] < cvalue[i]:
            norm_name.append(data.columns[i])
            norm_index.append(data.columns.get_loc(norm_name[j]))
            j=j+1
        else:
            notnorm_name.append(data.columns[i])
            notnorm_index.append(data.columns.get_loc(notnorm_name[k]))
            k=k+1
            
    return norm_index, norm_name, notnorm_index, notnorm_name



#Kruskal-Wallis test to discard features that are not important for classification of emotions

# Function to perform Kruskal-Wallis test for each feature
#creating a new dataframe to load the categorical values
efeats= feats_dfs
# Concatenate 'emotion' column to efeats
efeats['emotion'] = df['emotion'].values

# Function to perform Kruskal-Wallis test for each feature
def kruskal_wallis_1(efeats):
    results = []

    for feature in efeats.columns[:-1]:  # Exclude the 'emotion' column
        print(f"\nFeature: {feature}")
        groups = [group[1].tolist() for group in efeats.groupby('emotion')[feature]]
        for emotion, unique_values in zip(efeats['emotion'].unique(), groups):
            print(f"Emotion: {emotion}, Unique Values: {unique_values}")

        stat, p_value = kruskal(*groups)
        results.append({'Feature': feature, 'Statistic': stat, 'P-value': p_value})

    return pd.DataFrame(results)


#Kruskal-Wallis test result s
kw1_results = kruskal_wallis_1(efeats)


# Set the significance level
alpha = 0.01

# Select features with p-values less than alpha
selected_features = kw1_results[kw1_results['P-value'] < alpha]['Feature'].tolist()

# Filter features in feats_df based on selected features
filtered_krus = feats_dfs[selected_features]

feats_kwn = pd.DataFrame(scaler.fit_transform(filtered_krus), columns = filtered_krus.columns)



#define the SVM config for model
clf = svm.SVC(kernel='rbf')


'''
ICA and PCA test
for all selected by variance
for kw1 and kw2
for not normal and PCA normal

'''


#PCA settings
pca = PCA(n_components=100)

#scaler config for range -1 and 1
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

#########################


#PCA all
#scaler data for pca this step is not necessary but don't change the result
pca_all = pd.DataFrame(scaler.fit_transform(feats_kwn), columns = feats_kwn.columns)


#pca components extraction
PCA_comp_all = pca.fit_transform(pca_all)

#########################


#PCA not normal and PCA normal

#test distribution of data
norm_index_all, norm_name_all, notnorm_index_all, notnorm_name_all = distribuition(feats_kwn,95)


not_norm_all = feats_kwn.drop(norm_name_all,axis=1)

norm_all = feats_kwn.drop(notnorm_name_all,axis=1)

#pca components extraction

PCA_comp_all = pca.fit_transform(feats_kwn)

PCA_comp_norm_all = pca.fit_transform(norm_all)

PCA_comp_not_norm_all = pca.fit_transform(not_norm_all)




#########################



ica_all = pca_all #same scale for ica


ica_comp_all = []
 
ica_comp_norm_all = []
 
ica_comp_not_norm_all = []


#ICA components extraction

for i in range(1, 101):
    if i == 1:
        ica = FastICA(n_components = 2, algorithm= "deflation", whiten="unit-variance", fun='logcosh', fun_args={'alpha' : 1.0}, tol=1e-4, max_iter=500, w_init=None)   
        
        ica_comp_all.append(ica.fit_transform(ica_all))
        
        ica_comp_norm_all.append(ica.fit_transform(norm_all))
        
        ica_comp_not_norm_all.append(ica.fit_transform(not_norm_all))
        print(i)
    elif i % 10 == 0:
        print(i)

        ica = FastICA(n_components = i, algorithm= "deflation", whiten="unit-variance", fun='logcosh', fun_args={'alpha' : 1.0}, tol=1e-4, max_iter=500, w_init=None)   
        
        ica_comp_all.append(ica.fit_transform(ica_all))
        
        ica_comp_norm_all.append(ica.fit_transform(norm_all))
        
        ica_comp_not_norm_all.append(ica.fit_transform(not_norm_all))


#########################




pca_ica_matrix = []

for i in range(11):
    acc_pica = []
    for j in range(0, 110, 10):
        if i == 0 and j == 0 :
            acc_pica.append(0)
        elif i == 0:
            pi_feats = PCA_comp_norm_all[:,0:j-1]
            mean_acc = cross_val_score(clf, X=pi_feats, y=df.emotion.values, cv=10).mean()
            acc_pica.append(mean_acc)
        elif j == 0:
            pi_feats = ica_comp_all[i]
            mean_acc = cross_val_score(clf, X=pi_feats, y=df.emotion.values, cv=10).mean()
            acc_pica.append(mean_acc)
        else:
            pi_feats = (np.concatenate((ica_comp_all[i], PCA_comp_norm_all[:,0:j-1]),axis=1))
            mean_acc = cross_val_score(clf, X=pi_feats, y=df.emotion.values, cv=10).mean()
            acc_pica.append(mean_acc)
    pca_ica_matrix.append(acc_pica)

pca_ica_matrix_a = np.array(pca_ica_matrix)



pca_ica_all_matrix = []

for i in range(0,11):
    acc_pica = []
    for j in range(0, 110, 10):
        if i == 0 and j == 0 :
            acc_pica.append(0)
        elif i == 0:
            pi_feats = PCA_comp_all[:,0:j-1]
            mean_acc = cross_val_score(clf, X=pi_feats, y=df.emotion.values, cv=10).mean()
            acc_pica.append(mean_acc)
        elif j == 0:
            pi_feats = ica_comp_all[i]
            mean_acc = cross_val_score(clf, X=pi_feats, y=df.emotion.values, cv=10).mean()
            acc_pica.append(mean_acc)
        else:
            pi_feats = (np.concatenate((ica_comp_all[i], PCA_comp_all[:,0:j-1]),axis=1))
            mean_acc = cross_val_score(clf, X=pi_feats, y=df.emotion.values, cv=10).mean()
            acc_pica.append(mean_acc)   
    pca_ica_all_matrix.append(acc_pica)

pca_ica_matrix_all = np.array(pca_ica_all_matrix)




index_labels = np.flipud(np.arange(0, 109, 10))
column_labels = np.arange(0, 109, 10)


matrix_PI_dist_sep = pd.DataFrame(np.flipud(pca_ica_matrix_a), index=index_labels, columns=column_labels)
    

matrix_PI_all = pd.DataFrame(np.flipud(pca_ica_matrix_a), index=index_labels, columns=column_labels)    

import csv


csv_filenames = {
    'ravdess pca ica kw/norm_all.csv': norm_all,
    'ravdess pca ica kw/not_norm_all.csv': not_norm_all,
    'ravdess pca ica kw/pca_all.csv': PCA_comp_all,
    'ravdess pca ica kw/pca_normal.csv': PCA_comp_norm_all,
    'ravdess pca ica kw/pca_not_normal.csv': PCA_comp_not_norm_all,
    
}


feats_dfs.to_csv('ravdess pca ica/feats_stand.csv', sep=',', index=False, encoding='utf-8')
feats_df.to_csv('ravdess pca ica/feats_df.csv', sep=',', index=False, encoding='utf-8')
df.to_csv('ravdess pca ica/emotions.csv', sep=',', index=False, encoding='utf-8')



def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)


for filename, data in csv_filenames.items():
    save_to_csv(filename, data)


for i, n in enumerate(ica_comp_all):
    filename = f'ravdess pca ica kw/ica_com_all_{i+1}.csv'
    save_to_csv(filename, n)


for i, n in enumerate(ica_comp_norm_all):
    filename = f'ravdess pca ica kw/ica_norm_{i+1}.csv'
    save_to_csv(filename, n)


for i, n in enumerate(ica_comp_not_norm_all):
    filename = f'ravdess pca ica kw/ica_not_norm_{i+1}.csv'
    save_to_csv(filename, n)

