# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:36:49 2018

@author: Qing Gong
"""

import pandas as pd
import numpy as np
import string
import re

def feature_add(df):
    #Sentense count in each comment:
    #  '\n' can be used to count the number of sentences in each comment
    df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)

    df['total_length'] = df['comment_text'].apply(len)
    df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),
                                    axis=1)
    df['Cap_18_0'] = df['caps_vs_length'].apply(lambda ratio: 1*(ratio>0.18)+1*(ratio==0))
    df['Cap_40'] = df['caps_vs_length'].apply(lambda ratio: 1*(ratio>0.40))
    df['num_exclamation_marks'] = df['comment_text'].apply(lambda comment: comment.count('!'))
    df['num_question_marks'] = df['comment_text'].apply(lambda comment: comment.count('?'))
    df['num_equalsign'] = df['comment_text'].apply(lambda comment: comment.count('='))
    df['num_punctuation'] = df['comment_text'].apply(
        lambda comment: sum(comment.count(w) for w in string.punctuation))
    #df['num_symbols'] = df['comment_text'].apply(
    #    lambda comment: sum(comment.count(w) for w in '*&$%'))
    df['num_colon'] = df['comment_text'].apply(
        lambda comment: sum(comment.count(w) for w in ':'))
    df['num_words'] = df['comment_text'].apply(lambda comment: len(comment.split()))
    df['num_unique_words'] = df['comment_text'].apply(
        lambda comment: len(set(comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    df['spam'] = df['words_vs_unique'].apply(lambda ratio: 1*(ratio<0.3))
    df['num_smilies'] = df['comment_text'].apply(
        lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
    df['words_vs_sent'] = df['num_words'] / df['count_sent']
    df['punc_vs_sent'] = df['num_punctuation'] / df['count_sent']
    df['pvs_200_2000'] = df['punc_vs_sent'].apply(lambda ratio: 1*((ratio>200) & (ratio<2000)))
    
    df['punc_vs_words'] = df['num_punctuation'] / df['num_words']
    
#    df['count_stopwords'] = df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    df['mean_word_len'] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df['mean_sent_len'] = df['total_length'] / df['count_sent']
    
    df['count_letters'] = df["comment_text"].apply(lambda x: len(str(x)))
    df['unique_letter_in_sent'] = df['comment_text'].apply(lambda x: \
      np.mean([len(set(list(s)))/float(len(s)) for s in str(x).lower().split()])) / df['mean_sent_len']
    
    df['num_numbers'] = df["comment_text"].apply(lambda x: len(re.findall('(\d+)',x)))
    return df

# load file
train = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/input/train.csv')
test = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/input/test.csv')

print 'feature adding...'
train_fa = feature_add(train)
test_fa = feature_add(test)

print 'write to csv...'
train_fa.to_csv('E:/Google Drive/Kaggle/ToxicComment/other file/train_fa.csv', index=False)
test_fa.to_csv('E:/Google Drive/Kaggle/ToxicComment/other file/test_fa.csv', index=False)

print 'Done.'