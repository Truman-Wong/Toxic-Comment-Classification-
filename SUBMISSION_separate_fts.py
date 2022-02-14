# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 14:39:45 2018

@author: 09gq
"""

import pandas as pd
import numpy as np
from scipy.sparse import hstack, load_npz

from NBSVMClassifier import *

# load file
print 'load file ...'
train_tags = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/input/train.csv').iloc[:,2:]

subm = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/input/sample_submission.csv')

train_unigrams = load_npz('E:/Google Drive/Kaggle/ToxicComment/other file/train_unigrams.npz')
train_bigrams = load_npz('E:/Google Drive/Kaggle/ToxicComment/other file/train_bigrams.npz')
train_charngrams = load_npz('E:/Google Drive/Kaggle/ToxicComment/other file/train_charngrams.npz')

test_unigrams = load_npz('E:/Google Drive/Kaggle/ToxicComment/other file/test_unigrams.npz')
test_bigrams = load_npz('E:/Google Drive/Kaggle/ToxicComment/other file/test_bigrams.npz')
test_charngrams = load_npz('E:/Google Drive/Kaggle/ToxicComment/other file/test_charngrams.npz')

train_fa = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/other file/train_fa.csv')
test_fa = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/other file/test_fa.csv')
test_len = len(test_fa)

train_tfidf = hstack((train_unigrams, train_bigrams, train_charngrams))
test_tfidf = hstack((test_unigrams, test_bigrams, test_charngrams))
print 'load file completed.'

# major classes

FT_TOXIC = ['spam','caps_vs_length']
FT_OBSCENE = ['spam','Cap_18_0']
FT_INSULT = ['caps_vs_length']
# minor classes
FT_SEVERE_TOXIC = []
FT_THREAT = []
FT_IDENTITY_HATE = []

FTs = [FT_TOXIC, FT_SEVERE_TOXIC, FT_OBSCENE, FT_THREAT, FT_INSULT, FT_IDENTITY_HATE]

def Conc_to_csr(x, fa, fts):
    if not fts:
        return x.tocsr()
    else:
        return hstack((x, fa[fts])).tocsr()

#train_feature_TFIDF = train_tfidf.tocsr()
#test_feature_TFIDF = test_tfidf.tocsr()
#
#
#train_feature_toxic = hstack(train_tfidf, train_fa[FT_TOXIC]).tocsr()
#test_feature_toxic = hstack(test_tfidf, test_fa[FT_TOXIC]).tocsr()
#
#train_feature_obscene = hstack(train_tfidf, train_fa[FT_OBSCENE]).tocsr()
#test_feature_obscene = hstack(test_tfidf, test_fa[FT_OBSCENE]).tocsr()
#
#train_feature_insult = hstack(train_tfidf, train_fa[FT_INSULT]).tocsr()
#test_feature_insult = hstack(test_tfidf, test_fa[FT_INSULT]).tocsr()
#
#train_feature = []
#train_feature.append(train_feature_toxic)
#train_feature.append(train_feature_TFIDF)
#train_feature.append(train_feature_obscene)
#train_feature.append(train_feature_TFIDF)
#train_feature.append(train_feature_insult)
#train_feature.append(train_feature_TFIDF)
# 
print 'prepare model ...'
preds = np.zeros((test_len, 6))
# prepare model
NBSVM = NbSvmClassifier(alpha=1, C=4.0, dual=True, n_jobs=-1) # orig C=4
# fit the model
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for i, j in enumerate(label_cols):
    print 'work on label: '+ j    
    print 'concatenate train features ...'
    train_features = Conc_to_csr(train_tfidf, train_fa, FTs[i])
    print 'concatenate test features ...'
    test_features = Conc_to_csr(test_tfidf, test_fa, FTs[i])
    print 'train/fit ...'
    NBSVM.fit(train_features, train_tags[j])
    print 'predict ...'
    preds[:,i] = NBSVM.predict_proba(test_features)[:,1]

## Create the submission file
print 'creat submission file ...'
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('E:/Google Drive/Kaggle/ToxicComment/submission/submission.csv', index=False)
print 'Done.'