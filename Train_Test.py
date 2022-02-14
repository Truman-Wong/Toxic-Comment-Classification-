# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:20:01 2018

@author: Qing Gong
"""
#basics
import pandas as pd
import numpy as np
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from NBSVMClassifier import *

# load file
train = pd.read_csv('E:/Dropbox/Kaggle/ToxicComment/input/train.csv')
# add label 'clean'
#label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
label_cols = ['toxic']
train_tags = train[label_cols]
#train_tags=train.iloc[:,2:]

# Data preprocessing
import re, string
re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]
# tokenizer=tokenize, stop_words='english', 
vec = TfidfVectorizer(ngram_range=(1,2), analyzer='word',tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 ) # min_df default: 3

print 'tfidf fit and transform...'
#train_comt = train['comment_text']
# Spell Corrected comments
train_comt = pd.read_csv('E:/Dropbox/Kaggle/ToxicComment/other file/train_cr.csv')['comment_text']

trn_term_doc = vec.fit_transform(train_comt)

# feature adding
#print 'feature adding...'

train_fa = pd.read_csv('E:/Dropbox/Kaggle/ToxicComment/other file/train_fa.csv')
SELECTED_COLS = ['spam']
# 'num_words'
# 'caps_vs_length', 'num_exclamation_marks', 'words_vs_unique'
# 'num_punctuation', 'num_colon', 'num_unique_words', 'count_sent', 'num_exclamation_marks', 'num_unique_words'


#train_features = trn_term_doc
train_features = hstack((trn_term_doc, train_fa[SELECTED_COLS])).tocsr()
# prepare model
'split train and test set...'
NBSVM = NbSvmClassifier(alpha=0.1, C=4.0, dual=True, n_jobs=-1) # C default: 4
X_train, X_valid, y_train, y_valid = train_test_split(train_features, train_tags, train_size=0.7, random_state=666)

# fit the model
preds_valid = np.zeros((X_valid.shape[0], len(label_cols)))
auc_score = np.zeros(len(label_cols))
for i, j in enumerate(label_cols):
    print 'fit: '+ j
    NBSVM.fit(X_train, y_train[j])
    preds_valid[:,i] = NBSVM.predict_proba(X_valid)[:,1]
    auc_score[i] = roc_auc_score(y_valid[j], preds_valid[:,i])

# get auc score
auc_score # each class

auc_score_whole = roc_auc_score(y_valid, preds_valid)
auc_score_whole # whole data

#
y_valid_c = y_valid.as_matrix() # convert format
preds_valid_c = np.clip(preds_valid,a_min=1e-15,a_max=1-(1e-15)) # clip to range(0,1)
def sample_multilabel_logloss(y_true, y_pred):
    ''' Get log loss for multilabel classification, for each sample.
    return numpy array for n samples.'''
    loss = -y_true * np.log(y_pred) - (1-y_true) * np.log(1-y_pred)
    return loss.sum(axis=1)

#a = np.array([[0,0,0],[0,0,1]])
#b = np.ndarray([0.1,0,0])
#sample_multilabel_logloss(a, b)

valid_logloss = sample_multilabel_logloss(y_valid_c, preds_valid_c)

import seaborn as sns
sns.kdeplot(valid_logloss, label='log_loss') # KDE: kernal density estimate

sort_index = np.argsort(-valid_logloss)


for i in range(130,135):
    print 'Rank: '+str(i)
    large_loss_index = sort_index[i]
    print 'Log Loss:'+str(valid_logloss[large_loss_index]) +' true:'+ str(y_valid.iloc[large_loss_index].values)+' pred:'+str(preds_valid[large_loss_index])
    orig_index = y_valid.iloc[large_loss_index].name
    print train.iloc[orig_index]['comment_text']
#
#
i = 121
large_loss_index = sort_index[i]
print y_valid.iloc[large_loss_index]
print preds_valid[large_loss_index]
orig_index = y_valid.iloc[large_loss_index].name
print train.iloc[orig_index]['comment_text']
#
## spell check: original vs corrected
#i = 0
#large_loss_index = sort_index[i]
#orig_index = y_valid.iloc[large_loss_index].name
#print 'Original:'
#print(train.iloc[orig_index]['comment_text'])
#print 'Corrected:'
#print(train_cr.iloc[orig_index]['comment_text'])