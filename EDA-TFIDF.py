# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:24:28 2018

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


"""
TFIDF analysis
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string
re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()

### Unigrams -- TF-IDF 
# using settings recommended here for TF-IDF -- https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle

#some detailed description of the parameters
# min_df=10 --- ignore terms that appear lesser than 10 times 
# max_features=None  --- Create as many words as present in the text corpus
    # changing max_features to 10k for memmory issues
# analyzer='word'  --- Create features from words (alternatively char can also be used)
# ngram_range=(1,1)  --- Use only one word at a time (unigrams)
# strip_accents='unicode' -- removes accents
# use_idf=1,smooth_idf=1 --- enable IDF
# sublinear_tf=1   --- Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)


#temp settings to min=200 to facilitate top features section to run in kernals
#change back to min=10 to get better results    

#  w or w/o: tokenizer=tokenize, -- with punctuation, without punctuation
n = train.shape[0]
vec = TfidfVectorizer(min_df=200,  max_features=10000,  
            strip_accents='unicode', analyzer='word', ngram_range=(1,1),
            use_idf=1, smooth_idf=1, sublinear_tf=1,stop_words = 'english')
trn_term_doc = vec.fit_transform(train['comment_text'])
test_term_doc = vec.transform(test['comment_text'])

features = np.array(vec.get_feature_names())
train_tags = train.iloc[:,2:-1] # exclude Clean class
#get top n for unigrams
from TFIDF_analysis import *
tfidf_top_n_per_lass=top_feats_by_class(trn_term_doc, train_tags, features)

# plot
plt.figure(figsize=(16,22))
plt.suptitle("TF_IDF Top words per class(unigrams)",fontsize=20)
gridspec.GridSpec(4,2)
plt.subplot2grid((4,2),(0,0))
sns.barplot(tfidf_top_n_per_lass[0].feature.iloc[0:9],tfidf_top_n_per_lass[0].tfidf.iloc[0:9], color = 'blue')
plt.title("class : Toxic",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)

plt.subplot2grid((4,2),(0,1))
sns.barplot(tfidf_top_n_per_lass[1].feature.iloc[0:9],tfidf_top_n_per_lass[1].tfidf.iloc[0:9],color='red')
plt.title("class : Severe toxic",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(1,0))
sns.barplot(tfidf_top_n_per_lass[2].feature.iloc[0:9],tfidf_top_n_per_lass[2].tfidf.iloc[0:9],color='green')
plt.title("class : Obscene",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(1,1))
sns.barplot(tfidf_top_n_per_lass[3].feature.iloc[0:9],tfidf_top_n_per_lass[3].tfidf.iloc[0:9],color='black')
plt.title("class : Threat",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(2,0))
sns.barplot(tfidf_top_n_per_lass[4].feature.iloc[0:9],tfidf_top_n_per_lass[4].tfidf.iloc[0:9],color='cyan')
plt.title("class : Insult",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.subplot2grid((4,2),(2,1))
sns.barplot(tfidf_top_n_per_lass[5].feature.iloc[0:9],tfidf_top_n_per_lass[5].tfidf.iloc[0:9],color='yellow')
plt.title("class : Identity hate",fontsize=15)
plt.xlabel('Word', fontsize=12)
plt.ylabel('TF-IDF score', fontsize=12)


plt.show()


top_ft = []
for i in range(6):
    top_ft += tfidf_top_n_per_lass[i].feature.iloc[0:3].tolist()

top_ft = list(set(top_ft))

top_ft = [i.encode() for i in top_ft]
