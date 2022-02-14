# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:18:29 2018

@author: 09gq
"""

def cut_pred(x):
    if x > 0.998:
        return 1
    if x < 0.002:
        return 0
    return x

#map(cut_pred, [0.002,0.3,0.999])

cut_sub = pd.read_csv('E:/Dropbox/Kaggle/ToxicComment/submission/submission.csv')

cut_sub.iloc[:,1:] = cut_sub.iloc[:,1:].applymap(cut_pred)

cut_sub.to_csv('E:/Dropbox/Kaggle/ToxicComment/submission/cut_submission.csv', index=False)
