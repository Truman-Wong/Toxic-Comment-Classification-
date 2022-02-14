# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:31:08 2018

@author: 09gq
"""

#basics
import pandas as pd
import numpy as np
from scipy.sparse import hstack, load_npz

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from NBSVMClassifier import *

import matplotlib.pyplot as plt
import seaborn as sns

# load and concatenate file
print 'load and concatenate file ...'
label = 'toxic'
train_tags = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/input/train.csv')[label]

#trn_tfidf = load_npz('E:/Google Drive/Kaggle/ToxicComment/other file/trn_tfidf_word.npz')
# Spell Corrected TFIDF file
#trn_tfidf = load_npz('E:/Google Drive/Kaggle/ToxicComment/other file/trn_tfidf.npz')

train_unigrams = load_npz('E:/Google Drive/Kaggle/ToxicComment/other file/train_unigrams.npz')
train_bigrams = load_npz('E:/Google Drive/Kaggle/ToxicComment/other file/train_bigrams.npz')
train_charngrams = load_npz('E:/Google Drive/Kaggle/ToxicComment/other file/train_charngrams.npz')

train_fa = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/other file/train_fa.csv')
SELECTED_COLS = ['spam','caps_vs_length']


# train_unigrams, train_bigrams, train_charngrams, train_fa[SELECTED_COLS]
train_features = hstack((train_unigrams, train_bigrams, train_charngrams, train_fa[SELECTED_COLS])).tocsr()
#train_features = train_unigrams

# prepare model
print 'prepare model and Stratifield K-Fold ...'
NBSVM = NbSvmClassifier(alpha=1, C=4.0, dual=True, n_jobs=-1)

# Stratifield K-Fold
n_folds = 3
skf = StratifiedKFold(train_tags, n_folds=n_folds)

print 'Start Stratifield K-Fold train and test, label: ' + label


auc_score = np.zeros(n_folds)
i = -1
for train_index, test_index in skf:
    print 'New Fold...'
    i += 1
    X_train, X_test = train_features[train_index], train_features[test_index]
    y_train, y_test = train_tags[train_index], train_tags[test_index]
    NBSVM.fit(X_train, y_train)
    preds_valid = NBSVM.predict_proba(X_test)[:,1]
    auc_score[i] = roc_auc_score(y_test, preds_valid)
    print 'Fold AUC score: '+str(auc_score[i])

auc_score_avg = auc_score.mean()
print '---------Average AUC score: '+str(auc_score_avg)

# feature engineering: range analysis

#bad_count_whole = train_tags.value_counts()[1]
#both_count_whole = len(train_tags)
#bad_percent_whole = bad_count_whole / float(both_count_whole)
#print 'bad_percent_whole: '+str(bad_percent_whole)+' bad:'+str(bad_count_whole)+' both: '+str(both_count_whole)


#
## plot figure on selected feature
#Ana_Feat = 'num_question_marks'
#plt.figure()
#plt.title("Compare the selected feature.")
#sns.kdeplot(train_fa[train_tags == 0][Ana_Feat].squeeze(), label="Clean")
#sns.kdeplot(train_fa[train_tags == 1][Ana_Feat].squeeze(), label="Bad",shade=True,color='r')
#plt.legend()
#plt.ylabel('Density', fontsize=12)
#plt.xlabel(SELECTED_COLS[0], fontsize=12)
#plt.show()
#
#
#Ana_Feat = 'num_question_marks'
#range_whole = (7< train_fa[Ana_Feat]) & ( train_fa[Ana_Feat]<100)
#bad_count_range = train_tags[range_whole].value_counts()[1]
#bad_percent_range = bad_count_range / float(sum(range_whole))
#print 'bad_percent_range: '+str(bad_percent_range)+' bad:'+str(bad_count_range)+' both:' +str(sum(range_whole))
