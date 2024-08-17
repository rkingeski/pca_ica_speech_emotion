# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:33:48 2024

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

# Defina as variáveis onde os dados serão armazenados
feats_dfs = []
feats_df = []
df = []
norm_all = []
not_norm_all = []
PCA_comp_all = []
PCA_comp_norm_all = []
PCA_comp_not_norm_all = []
ica_comp_all = []
ica_comp_norm_all = []
ica_comp_not_norm_all = []

# Função para ler os dados de um arquivo CSV
def read_csv(filename):
    data = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data


labels = pd.DataFrame(read_csv('ravdess pca ica/emotions.csv'))
emotions = labels[1]
emotions = emotions.drop(0)

norm_all = pd.DataFrame(read_csv('ravdess pca ica/norm_all.csv'))
not_norm_all = pd.DataFrame(read_csv('ravdess pca ica/norm_all.csv'))
PCA_comp_all = pd.DataFrame(read_csv('ravdess pca ica/pca_all.csv'))
PCA_comp_norm_all = pd.DataFrame(read_csv('ravdess pca ica/pca_normal.csv'))
PCA_comp_not_norm_all = pd.DataFrame(read_csv('ravdess pca ica/pca_not_normal.csv'))

# Ler os dados dos arquivos ICA
for i in range(0,11):
    filename = f'ravdess pca ica/ica_com_all_{i+1}.csv'
    ica_comp_all.append(pd.DataFrame(read_csv(filename)))

for i in range(0,11):
    filename = f'ravdess pca ica/ica_norm_{i+1}.csv'
    ica_comp_norm_all.append(pd.DataFrame(read_csv(filename)))

for i in range(0,11):
    filename = f'ravdess pca ica/ica_not_norm_{i+1}.csv'
    ica_comp_not_norm_all.append(pd.DataFrame(read_csv(filename)))   




#########################


#define the SVM config for model
clf = svm.SVC(kernel='rbf', C=100)



#########################


PCA_comp_all = pd.DataFrame(read_csv('ravdess pca ica/pca_all.csv')).to_numpy()
PCA_comp_norm_all = pd.DataFrame(read_csv('ravdess pca ica/pca_normal.csv')).to_numpy()
PCA_comp_not_norm_all = pd.DataFrame(read_csv('ravdess pca ica/pca_not_normal.csv')).to_numpy()


# Ler os dados dos arquivos ICA
for i in range(0,11):
    filename = f'ravdess pca ica/ica_com_all_{i+1}.csv'
    ica_comp_all.append(pd.DataFrame(read_csv(filename)))

for i in range(0,11):
    filename = f'ravdess pca ica/ica_norm_{i+1}.csv'
    ica_comp_norm_all.append(pd.DataFrame(read_csv(filename)))

for i in range(0,11):
    filename = f'ravdess pca ica/ica_not_norm_{i+1}.csv'
    ica_comp_not_norm_all.append(pd.DataFrame(read_csv(filename)))   


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
                acc_pica.append(0)
            elif i == 0:
                pi_feats = pca[:,0:j-1]
                mean_acc = cross_val_score(clf, X=pi_feats, y=class_label, cv=10).mean()
                acc_pica.append(mean_acc)
            elif j == 0:
                pi_feats = ica[i]
                mean_acc = cross_val_score(clf, X=pi_feats, y=class_label, cv=10).mean()
                acc_pica.append(mean_acc)
            else:
                pi_feats = (np.concatenate((ica[i], pca[:,0:j-1]),axis=1))
                mean_acc = cross_val_score(clf, X=pi_feats, y=class_label, cv=10).mean()
                acc_pica.append(mean_acc)
        pca_ica_matrix.append(acc_pica)

    pca_ica_matrix = np.array(pca_ica_matrix)
    
    return pca_ica_matrix


#with distribuition sep
pca_ica_matrix_a = matrixacc(PCA_comp_norm_all, ica_comp_all, emotions, clf)

#without distribuition sep
pca_ica_matrix_all = matrixacc(PCA_comp_all, ica_comp_all, emotions, clf)



#################################################################################

#without KW

plotheat(pca_ica_matrix_a, 9, 'Heatmap_distribuition_separated_RAVDESS.png', True, 0.5)

plotheat(pca_ica_matrix_all, 9, 'Heatmap_distribuition_not_separated_RAVDESS.png', True, 0.5)



plotheat(pca_ica_matrix_a, 9, 'Heatmap_distribuition_separated2_RAVDESS.png' , False, 0)

plotheat(pca_ica_matrix_all, 9, 'Heatmap_distribuition_not_separated2_RAVDESS.png' , False, 0)


################################################################



##############################################################

#WITH KRUSKALL

###############################################


PCA_comp_all_kw = pd.DataFrame(read_csv('ravdess pca ica kw/pca_all.csv')).to_numpy()
PCA_comp_norm_all_kw = pd.DataFrame(read_csv('ravdess pca ica kw/pca_normal.csv')).to_numpy()
PCA_comp_not_norm_all_kw = pd.DataFrame(read_csv('ravdess pca ica kw/pca_not_normal.csv')).to_numpy()

ica_comp_all_kw = []

# Ler os dados dos arquivos ICA
for i in range(0,11):
    filename = f'ravdess pca ica kw/ica_com_all_{i+1}.csv'
    ica_comp_all_kw.append(pd.DataFrame(read_csv(filename)))



#########################################


# kw with distribuition sep
pca_ica_matrix_a_kw = matrixacc(PCA_comp_norm_all_kw, ica_comp_all_kw, emotions, clf)

# kw without distribuition sep
pca_ica_matrix_all_kw = matrixacc(PCA_comp_all_kw, ica_comp_all_kw, emotions, clf)

#################################################################################

#with KW

plotheat(pca_ica_matrix_a_kw, 9, 'Heatmap_distribuition_separated_RAVDESS_kw.png', True, 0.5)

plotheat(pca_ica_matrix_all_kw, 9, 'Heatmap_distribuition_not_separated_RAVDESS_kw.png', True, 0.5)



plotheat(pca_ica_matrix_a_kw, 9, 'Heatmap_distribuition_separated2_RAVDESS_kw.png' , False, 0)

plotheat(pca_ica_matrix_all_kw, 9, 'Heatmap_distribuition_not_separated2_RAVDESS_kw.png' , False, 0)




################################################################

#METRICS

#################################################################


def results(ica_feats, pca_feats, labels, n_folds):
    

    best_feats_sep = (np.concatenate((ica_feats, pca_feats),axis=1))
    pred = cross_val_predict(clf, X=best_feats_sep, y=labels.values, cv=n_folds)
    print(classification_report(labels, pred))
    metrics = classification_report(labels, pred, output_dict=True)
    metrics_df = pd.DataFrame(metrics).transpose()

    conf_mat = confusion_matrix(labels, pred)

    prec = cross_val_score(clf, X=best_feats_sep, y=labels.values, cv=n_folds, scoring='precision_macro')

    recall = cross_val_score(clf, X=best_feats_sep, y=labels.values, cv=n_folds, scoring='recall_macro')

    f1 = cross_val_score(clf, X=best_feats_sep, y=labels.values, cv=n_folds, scoring='f1_macro')

    acc = cross_val_score(clf, X=best_feats_sep, y=labels.values, cv=n_folds)

    return conf_mat, metrics_df, pred, prec, recall, f1, acc


# 100 ICA 50 PCA

conf_mat_sep, metrics_sep, pred_sep, prec_sep, recall_sep, f1_sep, acc_sep =  results(ica_comp_all[10], PCA_comp_norm_all[:,0:49], emotions, 10)


conf_mat_not_sep, metrics_sep, pred_not_sep, prec_not_sep, recall_not_sep, f1_not_sep, acc_not_sep =  results(ica_comp_all[10], PCA_comp_all[:,0:49], emotions, 10)




conf_mat_sep_kw, metrics_sep_kw, pred_sep_kw, prec_sep_kw, recall_sep_kw, f1_sep_kw, acc_sep_kw =  results(ica_comp_all_kw[10], PCA_comp_norm_all_kw[:,0:49], emotions, 10)


conf_mat_not_sep_kw, metrics_sep_kw, pred_not_sep_kw, prec_not_sep_kw, recall_not_sep_kw, f1_not_sep_kw, acc_not_sep_kw =  results(ica_comp_all_kw[10], PCA_comp_all_kw[:,0:49], emotions, 10)





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
#########################################################


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
plt.ylim(35, 80) 
plt.legend(title='Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.yticks(np.arange(35,81, 2.5))
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
plt.savefig('Ravdess_i100_p50.png', format='png')




##################

cm = conf_mat_sep_kw.astype('float') / conf_mat_sep_kw.sum(axis=1)[:, np.newaxis]

plt.figure()
clabels = np.unique(pred_sep_kw)
clabels = [label.capitalize() for label in clabels]

ax = sns.heatmap(cm*100, annot=True, cmap='Blues', fmt='.1f', xticklabels=clabels, yticklabels=clabels, cbar=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.xlabel('Predicted labels',y=1.05)
plt.ylabel('True labels')
plt.show()
plt.tight_layout()
plt.rcParams['savefig.dpi']=500
plt.savefig('Ravdess cm100_50_sep_kw.png', format='png')

cm = conf_mat_not_sep_kw.astype('float') / conf_mat_sep_kw.sum(axis=1)[:, np.newaxis]

plt.figure()

ax = sns.heatmap(cm*100, annot=True, cmap='Blues', fmt='.1f', xticklabels=clabels, yticklabels=clabels, cbar=False, cbar_kws={'label': 'Percentage (%)'})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
plt.tight_layout()
plt.rcParams['savefig.dpi']=500
plt.savefig('Ravdess cm100_50_not_sep_kw.png', format='png')
