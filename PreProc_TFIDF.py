# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:43:45 2018

@author: 09gq
"""
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

# load file
print 'load file ...'
train = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/input/train.csv')
train_len = train.shape[0]
#test = pd.read_csv('E:/Dropbox/Kaggle/ToxicComment/input/test.csv')

#train = pd.read_csv('E:/Dropbox/Kaggle/ToxicComment/other file/train_cr.csv')
#test = pd.read_csv('E:/Dropbox/Kaggle/ToxicComment/other file/test_cr.csv')

merge_clean = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/other file/merge_clean.csv')
# in case there is empty comment
print("filling NA with \"unknown\"")
merge_clean['comment_text'].fillna("unknown", inplace=True)

merge_cmt = merge_clean['comment_text']

print 'TF-IDF fit and transform ...'
import re, string
re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r' \1 ', s).split() # .split() will split string at space
# tokenizer is needed: add spaces to both sides of a punctuation, to treat the punctuation as a word
# example: 
#> s = ['hello! adaf#9.da @fad.eee, qqq!']s
#> re_tok.sub(r' \1 ', s)
#'hello !  adaf # 9 . da  @ fad . eee ,  qqq ! '

print 'fit unigrams ...'
tfv_uni = TfidfVectorizer(ngram_range=(1,1), tokenizer=tokenize, analyzer='word',
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )

tfv_uni.fit(merge_cmt)


print 'fit bigrams ...'
tfv_bi = TfidfVectorizer(ngram_range=(2,2), tokenizer=tokenize, analyzer='word',
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )

tfv_bi.fit(merge_cmt)


## character vectorizer
print 'fit chargrams ...'
tfv_char =  TfidfVectorizer(ngram_range=(1,4), max_features=30000, analyzer='char',
               min_df=100, strip_accents='unicode', use_idf=1, # stop_words = 'english',
               smooth_idf=1, sublinear_tf=1 )

tfv_char.fit(merge_cmt)

print 'train data: unigrams ...'
train_unigrams =  tfv_uni.transform(merge_cmt.iloc[:train_len])
print 'train data: bigrams ...'
train_bigrams =  tfv_bi.transform(merge_cmt.iloc[:train_len])
print 'train data: charngrams ...'
train_charngrams =  tfv_char.transform(merge_cmt.iloc[:train_len])

# concatenate train tfidf outputs
print 'train data: write to npz file ...'
#train_tfidf = hstack((train_unigrams,train_bigrams,train_charngrams)).tocsr()
#save_npz('E:/Google Drive/Kaggle/ToxicComment/other file/trn_tfidf.npz', train_tfidf)

save_npz('E:/Google Drive/Kaggle/ToxicComment/other file/train_unigrams.npz', train_unigrams)
save_npz('E:/Google Drive/Kaggle/ToxicComment/other file/train_bigrams.npz', train_bigrams)
save_npz('E:/Google Drive/Kaggle/ToxicComment/other file/train_charngrams.npz', train_charngrams)

# Test data
print 'test data: unigrams ...'
test_unigrams =  tfv_uni.transform(merge_cmt.iloc[train_len:])

print 'test data: bigrams ...'
test_bigrams =  tfv_bi.transform(merge_cmt.iloc[train_len:])

print 'test data: chargrams ...'
test_charngrams =  tfv_char.transform(merge_cmt.iloc[train_len:])

print 'test data: write to npz file ...'
#test_tfidf = hstack((test_unigrams,test_bigrams,test_charngrams)).tocsr()
#save_npz('E:/Google Drive/Kaggle/ToxicComment/other file/test_tfidf.npz', test_tfidf)

save_npz('E:/Google Drive/Kaggle/ToxicComment/other file/test_unigrams.npz', test_unigrams)
save_npz('E:/Google Drive/Kaggle/ToxicComment/other file/test_bigrams.npz', test_bigrams)
save_npz('E:/Google Drive/Kaggle/ToxicComment/other file/test_charngrams.npz', test_charngrams)

print 'Done.'