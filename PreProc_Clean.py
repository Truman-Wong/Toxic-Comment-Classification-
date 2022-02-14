# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:01:10 2018

@author: Qing Gong
"""

import pandas as pd
import numpy as np

from Corpus_Clean import *

# load file
train = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/input/train.csv')
test = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/input/test.csv')

merge=pd.concat([train.iloc[:,0:2],test.iloc[:,0:2]])
merge.reset_index(drop=True)

# corpus clean
print 'start cleaning comments ...'
merge['comment_text'].apply(lambda x: clean(x))

print 'write to csv file ...'
merge.to_csv('E:/Google Drive/Kaggle/ToxicComment/other file/merge_clean.csv', index=False)