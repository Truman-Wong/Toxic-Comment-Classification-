# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:56:27 2018

@author: Qing Gong
"""
#basics
import pandas as pd 
import numpy as np

# visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns

#stats
import scipy.stats as ss

# load file
train = pd.read_csv('E:/Dropbox/Kaggle/ToxicComment/input/train.csv')
test = pd.read_csv('E:/Dropbox/Kaggle/ToxicComment/input/test.csv')
subm = pd.read_csv('E:/Dropbox/Kaggle/ToxicComment/input/sample_submission.csv')

# add label 'clean'
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['clean'] = 1-train[label_cols].max(axis=1)
train.describe()

# data amounts
nrow_train=train.shape[0]
nrow_test=test.shape[0]
sum=nrow_train+nrow_test
print("       : train : test")
print("rows   :",nrow_train,":",nrow_test)
print("perc   :",round(nrow_train*100/sum),"   :",round(nrow_test*100/sum))

# check missing data
print("Check for missing values in Train dataset")
null_check=train.isnull().sum()
print(null_check)
print("Check for missing values in Test dataset")
null_check=test.isnull().sum()
print(null_check)


# plot
# classes
x=train.iloc[:,2:].sum(0) # sum columns
x.head(10)

plt.figure(figsize=(8,4))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("# per class")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Type ', fontsize=12)
#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()

# multi-tagging
x=train.iloc[:,2:].sum(1) # sum rows
x.head(10)  # list first 10 rows
x = x.value_counts()

#plot
plt.figure(figsize=(8,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Multiple tags per comment")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of tags ', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# check correlation among variables
temp_df = train.iloc[:,2:]

corr = temp_df.corr() #Compute pairwise correlation of columns, excluding NA/null values
plt.figure(figsize=(10,8))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True)
"""
The above plot indicates a pattern of co-occurance but Pandas's default Corr function
which uses Pearson correlation does not apply here, since the variables invovled are Categorical (binary) variables.
So, to find a pattern between two categorical variables we can use other tools like

1. Confusion matrix/Crosstab
2. Cramer's V Statistic
Cramer's V stat is an extension of the chi-square test where the extent/strength of association is also measured
"""

# Crosstab
# Since technically a crosstab between all 6 classes is impossible to vizualize, lets take a 
# look at toxic with other tags
main_col="toxic"
corr_mats=[]
for other_col in temp_df.columns[1:]:
    confusion_matrix = pd.crosstab(temp_df[main_col], temp_df[other_col])
    corr_mats.append(confusion_matrix)
out = pd.concat(corr_mats,axis=1,keys=temp_df.columns[1:])

#cell highlighting
out.style.highlight_min(axis=0)
out

"""
The above table represents the Crosstab/ consufion matix of Toxic comments with the other classes.

Some interesting observations:

1. A Severe toxic comment is always toxic
2. Other classes seem to be a subset of toxic barring a few exceptions
"""

#https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix/39266194
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#Checking for Toxic and Severe toxic for now
col1="toxic"
col2="severe_toxic"
confusion_matrix = pd.crosstab(temp_df[col1], temp_df[col2])
print("Confusion matrix between toxic and severe toxic:")
print(confusion_matrix)
new_corr=cramers_corrected_stat(confusion_matrix)
print("The correlation between Toxic and Severe toxic using Cramer's stat=",new_corr)


# Some examples
print("toxic:")
print(train[train.severe_toxic==1].iloc[3,1])
print("severe_toxic:")
print(train[train.severe_toxic==1].iloc[4,1])
print("Threat:")
print(train[train.threat==1].iloc[1,1])
print("Obscene:")
print(train[train.obscene==1].iloc[1,1])
print("identity_hate:")
print(train[train.identity_hate==1].iloc[4,1])



