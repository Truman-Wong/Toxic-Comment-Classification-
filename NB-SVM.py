# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 19:55:46 2018

@author: Qing Gong
"""
## Pre-processing
# import toolkit
import pandas as pd
import numpy as np
from scipy.sparse import hstack, load_npz

from NBSVMClassifier import *

# load file
print 'load file ...'
train_tags = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/input/train.csv').iloc[:,2:]

subm = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/input/sample_submission.csv')

trn_tfidf = load_npz('E:/Google Drive/Kaggle/ToxicComment/other file/trn_tfidf.npz')
test_tfidf = load_npz('E:/Google Drive/Kaggle/ToxicComment/other file/test_tfidf.npz')

train_fa = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/other file/train_fa.csv')
test_fa = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/other file/test_fa.csv')
test_len = len(test_fa)
## look at the data
#len(train),len(test)
#
#train.head()
#train[:][0:30]
#train['comment_text'][6]
#
#lens = train.comment_text.str.len()
#lens.hist()
#lens.mean(), lens.std(), lens.min(), lens.max()

# add label 'clean'
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
#train['clean'] = 1-train[label_cols].max(axis=1)
#train.describe()

## Building the model
# start by creating a bag of words representation, as a term document matrix

# Spell Corrected train and test comments


#trn_tfidf, test_tfidf
#
#trn_tfidf[0:2,].toarray()

print 'features concatenate ...'

#SELECTED_COLS = ['caps_vs_length', 'num_exclamation_marks', 'words_vs_unique']
#train_features = hstack((trn_tfidf, train_fa[SELECTED_COLS])).tocsr()
#test_features = hstack((test_tfidf, test_fa[SELECTED_COLS])).tocsr()

train_features = trn_tfidf
test_features = test_tfidf

preds = np.zeros((test_len, len(label_cols)))

# prepare model
NBSVM = NbSvmClassifier(alpha=1, C=4.0, dual=True, n_jobs=-1) # orig C=4

# fit the model

for i, j in enumerate(label_cols):
    print 'train/fit: ' + j
    NBSVM.fit(train_features, train_tags[j])
    print 'predict...'
    preds[:,i] = NBSVM.predict_proba(test_features)[:,1]

## Create the submission file
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('E:/Google Drive/Kaggle/ToxicComment/submission/submission.csv', index=False)
