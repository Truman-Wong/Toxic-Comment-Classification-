# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 23:43:57 2018

@author: Qing Gong
"""
#basics
import pandas as pd 
import numpy as np
from SpellCorrect import *

# load file
train = pd.read_csv('E:/Dropbox/Kaggle/ToxicComment/input/train.csv')
test = pd.read_csv('E:/Dropbox/Kaggle/ToxicComment/input/test.csv')

# choose popular toxic words

#toxiclist = ['fuck', 'cocksuck']
# from EDA-TFIDF: top 5 features each class
# eliminate 'going': common words, stop words
#toxiclist =['ass', 'gay', 'fucking', 'die', 'fuck', 'nigger', 'stupid', 'kill', 'bitch', 'suck', 'faggot', 'shit']
# top 10 features each class
#toxiclist = ['fucking', 'fuck', 'kill', 'bitch', 'shit', 'death', 'jew', 'idiot', 'll', 'dick', 'going', 'asshole', 'hope', 'ass', 'gay', 'stop', 'suck', 'cunt', 'don', 'like', 'die', 'nigger', 'stupid', 'faggot']
# top 3 features each class
toxiclist = ['gay', 'fucking', 'die', 'fuck', 'kill', 'bitch', 'faggot', 'shit']
import time

# get word frequency
print 'calculate word frequency...'
time_start = time.time()
train_comment_dict = word_freq(train['comment_text'])
time_count = time.time() - time_start

# get corrected comments
print 'get corrected comments: train...'
time_start = time.time()
tr_cr = ListCorrector(train['comment_text'], train_comment_dict)
train_cr = tr_cr.list_correct(10, toxiclist)
time_count = time.time() - time_start

print 'get corrected comments: test...'
test_comment_dict = word_freq(test['comment_text'])
ts_cr = ListCorrector(test['comment_text'], test_comment_dict)
test_cr = ts_cr.list_correct(10, toxiclist)

print 'write to .csv file...'
train_cr = pd.DataFrame(train_cr, columns = ['comment_text'])
train_cr.to_csv('E:/Dropbox/Kaggle/ToxicComment/other file/train_cr.csv', index=False)

test_cr = pd.DataFrame(test_cr, columns = ['comment_text'])
test_cr.to_csv('E:/Dropbox/Kaggle/ToxicComment/other file/test_cr.csv', index=False)