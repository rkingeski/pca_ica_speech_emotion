# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 00:38:29 2024

@author: rking
"""



import numpy as np
import pandas as pd
import audb
import opensmile
from scipy.stats import anderson
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, FastICA
from sklearn import preprocessing
  

# to load de Berlim databse using the audb library, you can install using pip install audb

db = audb.load('emodb') #load database

df = db.tables['emotion'].df #load table with emotions

new_label= {'anger':'Anger', 'boredom':'Boredom', 'disgust':'Disgust', 'happiness': 'Happiness',  'fear':'Fear', 'sadness':'Sadness', 'neutral':'Neutral'}

df.emotion = df.emotion.map(new_label)


#OpenSmile package to extract audio features, select the configurations and the group of features
smile = opensmile.Smile(
   feature_set=opensmile.FeatureSet.ComParE_2016,
   #feature_set=opensmile.FeatureSet.eGeMAPSv02,
   #feature_set=opensmile.FeatureSet.emobase,
   feature_level=opensmile.FeatureLevel.Functionals,
)


feats_df = smile.process_files(df.index) #load the features
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
#scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

#########################


#PCA all
#scaler data for pca this step is not necessary but don't chance the result

#pca components extraction
PCA_comp_all = pca.fit_transform(feats_dfs)

#########################


#PCA not normal and PCA normal

#test distribution of data
norm_index_all, norm_name_all, notnorm_index_all, notnorm_name_all = distribuition(feats_df,95)


not_norm_all = feats_dfs.drop(norm_name_all,axis=1)

norm_all = feats_dfs.drop(notnorm_name_all,axis=1)

#pca components extraction

PCA_comp_all = pca.fit_transform(feats_dfs)

PCA_comp_norm_all = pca.fit_transform(norm_all)

PCA_comp_not_norm_all = pca.fit_transform(not_norm_all)


#########################



ica_comp_all = []
 
ica_comp_norm_all = []
 
ica_comp_not_norm_all = []


#ICA components extraction

for i in range(1, 101):
    if i == 1:
        ica = FastICA(n_components = 2, algorithm= "deflation", whiten="unit-variance", fun='logcosh', fun_args={'alpha' : 1.0}, tol=1e-4, max_iter=500, w_init=None)   
        
        ica_comp_all.append(ica.fit_transform(feats_dfs))
        
        ica_comp_norm_all.append(ica.fit_transform(norm_all))
        
        ica_comp_not_norm_all.append(ica.fit_transform(not_norm_all))
        print(i)
    elif i % 10 == 0:
        print(i)

        ica = FastICA(n_components = i, algorithm= "deflation", whiten="unit-variance", fun='logcosh', fun_args={'alpha' : 1.0}, tol=1e-4, max_iter=500, w_init=None)   
        
        ica_comp_all.append(ica.fit_transform(feats_dfs))
        
        ica_comp_norm_all.append(ica.fit_transform(norm_all))
        
        ica_comp_not_norm_all.append(ica.fit_transform(not_norm_all))


#########################
pca_ica_matrix = []

for i in range(11):
    acc_pica = []
    for j in range(0, 110, 10):
        if i == 0 and j == 0 :
            #pi_feats = ica_comp_not_norm_all[i]
            #mean_acc = cross_val_score(clf, X=pi_feats, y=df.emotion.values, cv=10).mean()
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
            #pi_feats = ica_comp_not_norm_all[i]
            #mean_acc = cross_val_score(clf, X=pi_feats, y=df.emotion.values, cv=10).mean()
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




excel_filename = 'berlim_pca_ica.xlsx'


with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    
    pd.DataFrame(PCA_comp_all).to_excel(writer, sheet_name='pca_all', index=False)   
    pd.DataFrame(PCA_comp_norm_all).to_excel(writer, sheet_name='pca_normal', index=False)
    pd.DataFrame(PCA_comp_not_norm_all).to_excel(writer, sheet_name='pca_not_normal', index=False)

    
    for i, n in enumerate(ica_comp_all):
        ica_df= pd.DataFrame(n)
        ica_df.to_excel(writer, sheet_name=f'ica_com_all_{i+1}', index=False)
    
    for i, n in enumerate(ica_comp_norm_all):
        ica_df= pd.DataFrame(n)
        ica_df.to_excel(writer, sheet_name=f'ica_norm_{i+1}', index=False)
    
    for i, n in enumerate(ica_comp_not_norm_all):
        ica_df= pd.DataFrame(n)
        ica_df.to_excel(writer, sheet_name=f'ica_not_norm_{i+1}', index=False)
    
    
    matrix_PI_dist_sep.to_excel(writer, sheet_name='matrix_with_dist_sep', index=False)
    
    matrix_PI_all.to_excel(writer, sheet_name='matrix_without_dist_sep', index=False)


