# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:01:37 2024

@author: rking
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mstats
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict
import csv
import math
from sklearn import preprocessing
from scipy import stats

# Define the Excel filename
excel_filename = r'G:\Meu Drive\features_py\emodb scripts\berlim_pca_ica.xlsx'

# Read the Excel file
with pd.ExcelFile(excel_filename) as xls:
    # Read the sheets containing PCA data
    pca_all = pd.read_excel(xls, sheet_name='pca_all').to_numpy()
    pca_normal = pd.read_excel(xls, sheet_name='pca_normal').to_numpy()
    pca_not_normal = pd.read_excel(xls, sheet_name='pca_not_normal').to_numpy()
    
    # Create a dictionary to hold ICA data
    ica_data = {}
    ica_norm_data = {}
    ica_not_norm_data = {} 
    
    # Read the sheets containing ICA data
    for i in range(1, len(xls.sheet_names)):
        if xls.sheet_names[i].startswith('ica_com_all'):
            ica_data[i] = pd.read_excel(xls, sheet_name=xls.sheet_names[i])
        elif xls.sheet_names[i].startswith('ica_norm'):
            ica_norm_data[i] = pd.read_excel(xls, sheet_name=xls.sheet_names[i])
        elif xls.sheet_names[i].startswith('ica_not_norm'):
            ica_not_norm_data[i] = pd.read_excel(xls, sheet_name=xls.sheet_names[i])

ica = []
ica_norm = []
ica_not_norm = []

[ica.extend([v]) for k,v in ica_data.items()]
[ica_norm.extend([v]) for k,v in ica_norm_data.items()]
[ica_not_norm.extend([v]) for k,v in ica_not_norm_data.items()]

#########################

# Define the Excel filename
excel_filename = r'G:\Meu Drive\features_py\emodb scripts\berlim_pca_ica_kw.xlsx'

# Read the Excel file
with pd.ExcelFile(excel_filename) as xls:
    # Read the sheets containing PCA data
    pca_all_kw = pd.read_excel(xls, sheet_name='pca_all').to_numpy()
    pca_normal_kw = pd.read_excel(xls, sheet_name='pca_normal').to_numpy()
    pca_not_normal_kw = pd.read_excel(xls, sheet_name='pca_not_normal').to_numpy()
    emotions = pd.read_excel(xls, sheet_name='emotions')
    
    # Create a dictionary to hold ICA data
    ica_kw_data= {}
    ica_norm_kw_data = {}
    ica_not_norm_kw_data = {}
    
    # Read the sheets containing ICA data
    for i in range(1, len(xls.sheet_names)):
        if xls.sheet_names[i].startswith('ica_com_all'):
            ica_kw_data[i] = pd.read_excel(xls, sheet_name=xls.sheet_names[i])
        elif xls.sheet_names[i].startswith('ica_norm'):
            ica_norm_kw_data[i] = pd.read_excel(xls, sheet_name=xls.sheet_names[i])
        elif xls.sheet_names[i].startswith('ica_not_norm'):
            ica_not_norm_kw_data[i] = pd.read_excel(xls, sheet_name=xls.sheet_names[i])


ica_kw = []
ica_norm_kw = []
ica_not_norm_kw = []

[ica_kw.extend([v]) for k,v in ica_kw_data.items()]
[ica_norm_kw.extend([v]) for k,v in ica_norm_kw_data.items()]
[ica_not_norm_kw.extend([v]) for k,v in ica_not_norm_kw_data.items()]

#########################



#define the SVM config for model
clf = svm.SVC(kernel='rbf', C=100)

'''
scaler = preprocessing.StandardScaler().fit(pca_all)
pca_all = scaler.fit_transform(pca_all)
scaler = preprocessing.StandardScaler().fit(pca_normal)
pca_normal = scaler.fit_transform(pca_normal)
scaler = preprocessing.StandardScaler().fit(pca_not_normal)
pca_not_normal = scaler.fit_transform(pca_not_normal)

scaler = preprocessing.StandardScaler().fit(pca_all_kw)
pca_all_kw = scaler.fit_transform(pca_all_kw)
scaler = preprocessing.StandardScaler().fit(pca_normal_kw)
pca_normal_kw = scaler.fit_transform(pca_normal_kw)
scaler = preprocessing.StandardScaler().fit(pca_not_normal_kw)
pca_not_normal_kw = scaler.fit_transform(pca_not_normal_kw)
'''



#########################

'''
accuracy_pca_all = []

accuracy_pca_norm_all = []

accuracy_pca_not_norm_all = []


for i in range(0, 101, 10):
    
    if i == 0:
    
        scores_pca_all = cross_val_score(clf, X=pca_all[:,0:1], y=emotions, cv=10)
       
        scores_pca_norm_all = cross_val_score(clf, X=pca_normal[:,0:1], y=emotions, cv=10)
        
        scores_pca_not_norm_all = cross_val_score(clf, X=pca_not_normal[:,0:1], y=emotions, cv=10)
       
        accuracy_pca_all.append(scores_pca_all)
        
        accuracy_pca_norm_all.append(scores_pca_norm_all)
        
        accuracy_pca_not_norm_all.append(scores_pca_not_norm_all)
        
    else:
        
        scores_pca_all = cross_val_score(clf, X=pca_all[:,0:i-1], y=emotions, cv=10)
       
        scores_pca_norm_all = cross_val_score(clf, X=pca_normal[:,0:i-1], y=emotions, cv=10)
        
        scores_pca_not_norm_all = cross_val_score(clf, X=pca_not_normal[:,0:i-1], y=emotions, cv=10)
       
        accuracy_pca_all.append(scores_pca_all)
        
        accuracy_pca_norm_all.append(scores_pca_norm_all)
        
        accuracy_pca_not_norm_all.append(scores_pca_not_norm_all)
        


#plot the mean accuracy versus number of principal components 
    
mean_accuracy_pca_all = np.array([np.mean(lines) for lines in accuracy_pca_all])

mean_accuracy_pca_norm_all = np.array([np.mean(lines) for lines in accuracy_pca_norm_all])

mean_accuracy_pca_not_norm_all = np.array([np.mean(lines) for lines in accuracy_pca_not_norm_all])

number_pca = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

#plt.figure(figsize=(14, 8))
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

plt.figure(facecolor='white')
plt.grid(which='major', linestyle='--', linewidth=0.5, color='gray')
plt.xticks(number_pca)

plt.plot(number_pca, mean_accuracy_pca_all, 'o-', label='Mean Accu All Features', markersize=8)
plt.plot(number_pca, mean_accuracy_pca_norm_all, '^-', label='Mean Accu Normal Features', markersize=8)
plt.plot(number_pca, mean_accuracy_pca_not_norm_all, 'v-', label='Mean Accu Non-Normal Features', markersize=8)


plt.xlabel('Number of Components')
plt.ylabel('Accuracy of Model')
plt.title('EmoDB PCA')


plt.legend()

plt.tight_layout()
plt.show()

plt.rcParams['savefig.dpi']=600
plt.savefig('Emodb_PCA.png', format='png')




#########################

accuracy_ica_all = []

accuracy_ica_norm_all = []

accuracy_ica_not_norm_all = []


for i in range(len(ica[:])):
    
    scores_ica_all = cross_val_score(clf, X=ica[i], y=emotions, cv=10)
   
    scores_ica_norm_all = cross_val_score(clf, X=ica_norm[i], y=emotions, cv=10)
    
    scores_ica_not_norm_all = cross_val_score(clf, X=ica_not_norm[i], y=emotions, cv=10)
   
    accuracy_ica_all.append(scores_ica_all)
    
    accuracy_ica_norm_all.append(scores_ica_norm_all)
    
    accuracy_ica_not_norm_all.append(scores_ica_not_norm_all)
    
    
    print(i)


#plot the mean accuracy versus number of independent components 
    
mean_accuracy_ica_all = np.array([np.mean(lines) for lines in accuracy_ica_all])

std_accuracy_ica_all = np.array([np.std(lines) for lines in accuracy_ica_all])

mean_accuracy_ica_norm_all = np.array([np.mean(lines) for lines in accuracy_ica_norm_all])

std_accuracy_ica_norm_all = np.array([np.std(lines) for lines in accuracy_ica_norm_all])

mean_accuracy_ica_not_norm_all = np.array([np.mean(lines) for lines in accuracy_ica_not_norm_all])

std_accuracy_ica_not_norm_all = np.array([np.std(lines) for lines in accuracy_ica_not_norm_all])


number_ica = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


#plt.figure(figsize=(14, 8))
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

plt.figure(facecolor='white')
plt.grid(which='major', linestyle='--', linewidth=0.5, color='gray')
plt.xticks(number_ica)

plt.plot(number_ica, mean_accuracy_ica_all, 'o-', label='Mean Accu All Features', markersize=8)
plt.plot(number_ica, mean_accuracy_ica_norm_all, '^-', label='Mean Accu Normal Features', markersize=8)
plt.plot(number_ica, mean_accuracy_ica_not_norm_all, 'v-', label='Mean Accu Non-Normal Features', markersize=8)


plt.xlabel('Number of Components')
plt.ylabel('Accuracy of Model')
plt.title('EmoDB ICA')


plt.legend()

plt.tight_layout()
plt.show()

plt.rcParams['savefig.dpi']=600
plt.savefig('Emodb_ICA.png', format='png')

'''
####################



###########################################

def plotheat(matrix, n_scale, name_of_file, labels, linew):

    plt.figure(facecolor='white')

    index_labels = np.flipud(np.arange(0, 109, 10))
    column_labels = np.arange(0, 109, 10)


    matrix_PI = pd.DataFrame(np.flipud(matrix)*100, index=index_labels, columns=column_labels)

    min_value = math.floor(np.min(matrix[matrix != 0]*100))
    max_value = math.ceil(np.max(matrix*100))
    
    htmap =sns.heatmap(matrix_PI, cmap="Spectral", mask=matrix_PI== 0.0 ,annot=labels, linewidth=linew,vmin=min_value, vmax=max_value, fmt=".1f", cbar_kws={'label': 'Percentage %'})
    plt.xlabel('Principal Components')
    plt.ylabel('Independent Components')
    #plt.title('')
    cbar = htmap.collections[0].colorbar  # Get the color bar object
    cbar.set_ticks(np.linspace(min_value, max_value, num=n_scale)) 
    plt.show()

    plt.rcParams['savefig.dpi']=600
    plt.savefig(name_of_file, format='png')



#########################################
# function to generate the matrix of accuracy
#########################################
def matrixacc(pca,ica, class_label, clf):
    
    pca_ica_matrix = []

    for i in range(11):
        acc_pica = []
        for j in range(0, 110, 10):
            if i == 0 and j == 0 :
                #pi_feats = ica_comp_not_norm_all[i]
                #mean_acc = cross_val_score(clf, X=pi_feats, y=emotions, cv=10).mean()
                acc_pica.append(0)
            elif i == 0:
                pi_feats = pca[:,0:j-1]
                mean_acc = cross_val_score(clf, X=pi_feats, y=class_label.values.ravel(), cv=10).mean()
                acc_pica.append(mean_acc)
            elif j == 0:
                pi_feats = ica[i]
                mean_acc = cross_val_score(clf, X=pi_feats, y=class_label.values.ravel(), cv=10).mean()
                acc_pica.append(mean_acc)
            else:
                pi_feats = (np.concatenate((ica[i], pca[:,0:j-1]),axis=1))
                mean_acc = cross_val_score(clf, X=pi_feats, y=class_label.values.ravel(), cv=10).mean()
                acc_pica.append(mean_acc)
        pca_ica_matrix.append(acc_pica)

    pca_ica_matrix = np.array(pca_ica_matrix)
    
    return pca_ica_matrix

clf = svm.SVC(kernel='rbf', C=100)

#with distribuition sep
#pca_ica_matrix_a = matrixacc(pca_normal, ica, emotions, clf)

#without distribuition sep
#pca_ica_matrix_all = matrixacc(pca_all, ica, emotions, clf)


#################################################################################

#without KW
'''
plotheat(pca_ica_matrix_a, 9, 'Heatmap_distribuition_separated_EmoDB.png', True, 0.5)

plotheat(pca_ica_matrix_all, 9, 'Heatmap_distribuition_not_separated_EmoDB.png', True, 0.5)



plotheat(pca_ica_matrix_a, 9, 'Heatmap_distribuition_separated2_EmoDB.png' , False, 0)

plotheat(pca_ica_matrix_all, 9, 'Heatmap_distribuition_not_separated2_EmoDB.png' , False, 0)





##############################################################

#WITH KRUSKALL

###############################################



# kw with distribuition sep
pca_ica_matrix_a_kw = matrixacc(pca_normal_kw, ica_kw, emotions, clf)

# kw without distribuition sep
pca_ica_matrix_all_kw = matrixacc(pca_all_kw, ica_kw, emotions, clf)

#################################################################################

#with KW

plotheat(pca_ica_matrix_a_kw, 9, 'Heatmap_distribuition_separated_EmoDB_kw.png', True, 0.5)

plotheat(pca_ica_matrix_all_kw, 9, 'Heatmap_distribuition_not_separated_EmoDB_kw.png', True, 0.5)



plotheat(pca_ica_matrix_a_kw, 9, 'Heatmap_distribuition_separated2_EmoDB_kw.png' , False, 0)

plotheat(pca_ica_matrix_all_kw, 9, 'Heatmap_distribuition_not_separated2_EmoDB_kw.png' , False, 0)


#####################

#ica not normal test

# kw with distribuition sep
pca_ica_matrix_a_kw_t = matrixacc(pca_normal_kw, ica_not_norm_kw, emotions, clf)



plotheat(pca_ica_matrix_a_kw_t, 9, 'Heatmap_distribuition_separated_test_EmoDB_kw.png', True, 0.5)



plotheat(pca_ica_matrix_a_kw_t, 9, 'Heatmap_distribuition_separated2_test_EmoDB_kw.png' , False, 0)





#################################################################################

#with KW

plotheat(pca_ica_matrix_a_kw, 9, 'Heatmap_distribuition_separated_EmoDB_kw.png', True, 0.5)

plotheat(pca_ica_matrix_all_kw, 9, 'Heatmap_distribuition_not_separated_EmoDB_kw.png', True, 0.5)



plotheat(pca_ica_matrix_a_kw, 9, 'Heatmap_distribuition_separated2_EmoDB_kw.png' , False, 0)

plotheat(pca_ica_matrix_all_kw, 9, 'Heatmap_distribuition_not_separated2_EmoDB_kw.png' , False, 0)



'''

################################################################

#barplot

#################################################################

def results(ica_feats, pca_feats, labels, n_folds):
    

    best_feats_sep = (np.concatenate((ica_feats, pca_feats),axis=1))
    pred = cross_val_predict(clf, X=best_feats_sep, y=labels.values.ravel(), cv=n_folds)
    print(classification_report(labels.values.ravel(), pred,zero_division=0))
    metrics = classification_report(labels.values.ravel(), pred, output_dict=True)
    metrics_df = pd.DataFrame(metrics).transpose()

    conf_mat = confusion_matrix(labels, pred)

    prec = cross_val_score(clf, X=best_feats_sep, y=labels.values.ravel(), cv=n_folds, scoring='precision_macro')

    recall = cross_val_score(clf, X=best_feats_sep, y=labels.values.ravel(), cv=n_folds, scoring='recall_macro')

    f1 = cross_val_score(clf, X=best_feats_sep, y=labels.values.ravel(), cv=n_folds, scoring='f1_macro')

    acc = cross_val_score(clf, X=best_feats_sep, y=labels.values.ravel(), cv=n_folds)

    return conf_mat, metrics_df, pred, prec, recall, f1, acc


# 

conf_mat_sep, metrics_sep, pred_sep, prec_sep, recall_sep, f1_sep, acc_sep =  results(ica[10], pca_normal[:,0:70], emotions, 10)


conf_mat_not_sep, metrics_not_sep, pred_not_sep, prec_not_sep, recall_not_sep, f1_not_sep, acc_not_sep =  results(ica[10], pca_all[:,0:70], emotions, 10)




conf_mat_sep_kw, metrics_sep_kw, pred_sep_kw, prec_sep_kw, recall_sep_kw, f1_sep_kw, acc_sep_kw =  results(ica_kw[10], pca_normal_kw[:,0:70], emotions, 10)


conf_mat_not_sep_kw, metrics_not_sep_kw, pred_not_sep_kw, prec_not_sep_kw, recall_not_sep_kw, f1_not_sep_kw, acc_not_sep_kw =  results(ica_kw[10], pca_all_kw[:,0:70], emotions, 10)





prec_sep = prec_sep.reshape(-1, 1)
recall_sep = recall_sep.reshape(-1, 1)
f1_sep = f1_sep.reshape(-1, 1)
acc_sep = acc_sep.reshape(-1, 1)
prec_not_sep = prec_not_sep.reshape(-1, 1)
recall_not_sep = recall_not_sep.reshape(-1, 1)
f1_not_sep = f1_not_sep.reshape(-1, 1)
acc_not_sep = acc_not_sep.reshape(-1, 1)


prec_sep_kw = prec_sep_kw.reshape(-1, 1)
recall_sep_kw = recall_sep_kw.reshape(-1, 1)
f1_sep_kw = f1_sep_kw.reshape(-1, 1)
acc_sep_kw = acc_sep_kw.reshape(-1, 1)

prec_not_sep_kw = prec_not_sep_kw.reshape(-1, 1)
recall_not_sep_kw = recall_not_sep_kw.reshape(-1, 1)
f1_not_sep_kw = f1_not_sep_kw.reshape(-1, 1)
acc_not_sep_kw = acc_not_sep_kw.reshape(-1, 1)




combined_data = np.concatenate((prec_sep, recall_sep, f1_sep, acc_sep, prec_not_sep, recall_not_sep, f1_not_sep, acc_not_sep, prec_sep_kw, recall_sep_kw, f1_sep_kw, acc_sep_kw, prec_not_sep_kw, recall_not_sep_kw, f1_not_sep_kw, acc_not_sep_kw), axis=1)


column_names = ['prec_sep', 'recall_sep', 'f1_sep', 'acc_sep', 'prec_not_sep', 'recall_not_sep', 'f1_not_sep', 'acc_not_sep','prec_sep_kw', 'recall_sep_kw', 'f1_sep_kw', 'acc_sep_kw', 'prec_not_sep_kw', 'recall_not_sep_kw', 'f1_not_sep_kw','acc_not_sep_kw']


labels = ['Precision', 'Recall', 'F1 Score','Accuracy']


final_results = pd.DataFrame(combined_data, columns = column_names)

mean_values = final_results.mean()

std = final_results.std()

ci_up = []
ci_low = []
confid =  []
confidvalue = []

for i in range(0,16):
    ci_up.append(mean_values[i] + (std[i]/np.sqrt(10)))
    ci_low.append(mean_values[i] - (std[i]/np.sqrt(10)))
    confid.append((ci_low[i],ci_up[i]))
    confidvalue.append(ci_up[i]-ci_low[i])


labels2 = ['with dist', 'without dist', 'with dist - kw', 'without dist - kw']

def repeatl(lab):
    if not lab:  # Check if the list is empty
        return []  # Return an empty list if the input list is empty
    else:
        first = lab[0]  # Get the first element of the list
        repeat = [first] * 4  # Repeat the first element four times
        rest = lab[1:]  # Get the rest of the elements
        return repeat + repeatl(rest)  # Concatenate the repeated and original elements

labels_group = repeatl(labels2)

confid2 = pd.DataFrame(confid)

meanv = pd.DataFrame(mean_values.values) 
results_plot = pd.concat([pd.DataFrame(mean_values.values),pd.DataFrame(confidvalue),pd.DataFrame(labels * 4),pd.DataFrame(labels_group)],axis=1)
results_plot.columns = ['mean','confidence95','metric','group']

#ax = sns.barplot(data = results_plot, hue="group", y="mean", x="metric")



#########################################################
# BOXPLOT
##########################


final_test = final_results.transpose()*100
final_test = final_test.reset_index()
results_test = pd.concat([final_test,pd.DataFrame(labels * 4,columns=['Metric']),pd.DataFrame(labels_group,columns=['Groups'])],axis=1)



# Melt the DataFrame to bring 'Metric' as a variable
melted_df = pd.melt(results_test, id_vars=['Groups', 'Metric'], var_name='Accuracy', value_name='Value')

# Drop rows where 'Value' is not numeric (e.g., column names)
melted_df = melted_df[melted_df['Value'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]

# Convert 'Value' column to numeric
melted_df['Value'] = pd.to_numeric(melted_df['Value'])

# Convert 'Metric' column to categorical
melted_df['Metric'] = pd.Categorical(melted_df['Metric'], categories=['Precision', 'Recall', 'F1 Score', 'Accuracy'])


group_order = ['without dist', 'with dist', 'without dist - kw', 'with dist - kw']
grouporder = ['Without distribuition separation' , 'With distribuition separation', 'Without distribuition separation - KW', 'With distribuition separation - KW']

group_rename = {'without dist':'Without distribuition separation' , 'with dist': 'With distribuition separation', 'without dist - kw': 'Without distribuition separation - KW', 'with dist - kw': 'With distribuition separation - KW'}

melted_df['Groups'] = melted_df['Groups'].map(group_rename)

# Plot the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=melted_df, x='Metric', y='Value', hue='Groups',showmeans=True, hue_order=grouporder,palette=['#AED6F1', '#ABEBC6', '#F9E79F', '#F5B7B1'], saturation=2.0,
            meanprops={"marker":"x", "markerfacecolor":"grey", "markeredgecolor":"black", "markersize":"7"})
plt.xlabel('Metrics')
plt.ylabel('Accuracy(%)')
#plt.title('Box Plot of Accuracy by Metric and Group')
plt.grid(axis='y', linestyle='--', color='grey', linewidth=0.5)
plt.ylim(60, 100) 
#plt.legend(title='Groups')
plt.legend(title='Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.yticks(np.arange(60,101, 2.5))
#plt.box(on=True)
plt.tight_layout()

# Add a box around the plot area
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

plt.gca().spines['top'].set_color('grey')  # Top spine color
plt.gca().spines['right'].set_color('grey')  # Right spine color
plt.gca().spines['bottom'].set_color('grey')  # Bottom spine color
plt.gca().spines['left'].set_color('grey')  # Left spine color

plt.show()
plt.rcParams['savefig.dpi']=600
plt.savefig('Emodb_i100_p70.png', format='png')

'''
barWidth = 0.15





bars1 = results_plot.loc[results_plot['group'] == 'without dist', 'mean'].tolist()

bars2 = results_plot.loc[results_plot['group'] == 'with dist', 'mean'].tolist()

bars3 = results_plot.loc[results_plot['group'] == 'without dist - kw', 'mean'].tolist()

bars4 = results_plot.loc[results_plot['group'] == 'with dist - kw', 'mean'].tolist()

yerr1 = results_plot.loc[results_plot['group'] == 'without dist', 'confidence95'].tolist()

yerr2 = results_plot.loc[results_plot['group'] == 'with dist', 'confidence95'].tolist()

yerr3 = results_plot.loc[results_plot['group'] == 'without dist - kw', 'confidence95'].tolist()

yerr4 = results_plot.loc[results_plot['group'] == 'with dist - kw', 'confidence95'].tolist()


bars1 = list(map(lambda x: 100*x, bars1))

bars2 = list(map(lambda x: 100*x, bars2))

bars3 = list(map(lambda x: 100*x, bars3))

bars4 = list(map(lambda x: 100*x, bars4))

yerr1 = list(map(lambda x: 100*x, yerr1))

yerr2 = list(map(lambda x: 100*x, yerr2))

yerr3 = list(map(lambda x: 100*x, yerr3))

yerr4 = list(map(lambda x: 100*x, yerr4))

r1 = np.arange(len(bars1))
r2 = [x + 0.03 + barWidth for x in r1]
r3 = [x + 0.03 + barWidth for x in r2]
r4 = [x + 0.03 + barWidth for x in r3]


plt.figure()
#sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = '#AED6F1', edgecolor = 'black', yerr=yerr1, capsize=4, label='Without distribuition separation')
 
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = '#ABEBC6', edgecolor = 'black', yerr=yerr2, capsize=4, label='With distribuition separation')

plt.bar(r3, bars3, width = barWidth, color = '#F9E79F', edgecolor = 'black', yerr=yerr3, capsize=4, label='Without distribuition separation - KW')

plt.bar(r4, bars4, width = barWidth, color = '#F5B7B1', edgecolor = 'black', yerr=yerr4, capsize=4, label='With distribuition separation - KW')

# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], labels)
plt.xlabel('Metric')
plt.ylabel('Percentage %')
plt.ylim(70, 94) 
plt.grid(True, linestyle='--', color='gray', linewidth=0.6)
plt.legend()
plt.tight_layout()
plt.rcParams['savefig.dpi']=500
plt.savefig('emodb_100_70.png', format='png')



'''

#######

cm = 100*conf_mat_sep_kw.astype('float') / conf_mat_sep_kw.sum(axis=1)[:, np.newaxis]

plt.figure()
clabels = np.unique(pred_sep_kw)
clabels = [label.capitalize() for label in clabels]

ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=clabels, yticklabels=clabels, cbar=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.xlabel('Predicted labels',y=1.05)
plt.ylabel('True labels')
plt.show()
plt.tight_layout()
plt.rcParams['savefig.dpi']=600
plt.savefig('Emodb cm100_70_sep_kw.png', format='png')

cm = 100*conf_mat_not_sep_kw.astype('float') / conf_mat_sep_kw.sum(axis=1)[:, np.newaxis]

plt.figure()

ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=clabels, yticklabels=clabels, cbar=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
plt.tight_layout()
plt.rcParams['savefig.dpi']=600
plt.savefig('Emodb cm100_70_not_sep_kw.png', format='png')


'''
# Calcular a média e o desvio padrão de cada grupo
mean1 = np.mean(acc_sep_kw)
mean2 = np.mean(acc_not_sep)
std1 = np.std(acc_sep_kw, ddof=1)
std2 = np.std(acc_not_sep, ddof=1)

# Calcular o tamanho do efeito (d de Cohen)
pooled_std = np.sqrt((std1**2 + std2**2) / 2)
effect_size = (mean1 - mean2) / pooled_std

print(f'Tamanho do Efeito (d de Cohen): {effect_size:.2f}')
'''