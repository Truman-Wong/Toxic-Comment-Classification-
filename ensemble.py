# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:42:13 2018

@author: 09gq
"""

import pandas as pd
import numpy as np

NBSVM = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/submission/submission.csv')
#NBSVM_word = pd.read_csv('E:/Dropbox/Kaggle/ToxicComment/submission/submission7.csv')
#NBSVM_char = pd.read_csv('E:/Dropbox/Kaggle/ToxicComment/submission/submission8.csv')
gruglo = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/submission/GRU_glove_sub.csv')


#hight = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/Others_sub/hight_of_blending 9860.csv')
#ave = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/Others_sub/Toxic Avenger(LSTM with BN + NB-SVM + LR on Conv AI) 9823.csv')
#onemore = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/Others_sub/one_more_blend 9865.csv')
#bldall = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/Others_sub/blend_it_all 9867.csv')


#mine = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/submission/submission10.csv')
blend = NBSVM.copy()

cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for i in cols[1:]:
    blend[i] = (NBSVM[i] + gruglo[i]) / 2 # + hight[i] 

blend.to_csv('E:/Google Drive/Kaggle/ToxicComment/submission/blend_NBSVM_GRUGLOVE.csv', index=False)

print('blend Done.')