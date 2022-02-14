# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 09:42:25 2018

@author: 09gq
"""
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# analyze subject
label = 'identity_hate'
Ana_Feat = 'unique_letter_in_sent'

print 'load file ...'
train_fa = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/other file/train_fa.csv')
train_tags = pd.read_csv('E:/Google Drive/Kaggle/ToxicComment/input/train.csv')[label]

## plot figure on selected feature
print 'plot ...'
plt.figure()
plt.title("Compare the selected feature.")
sns.kdeplot(train_fa[train_tags == 0][Ana_Feat].squeeze(), label="Clean")
sns.kdeplot(train_fa[train_tags == 1][Ana_Feat].squeeze(), label="Bad",shade=True,color='r')
plt.legend()
plt.ylabel('Density', fontsize=12)
plt.xlabel(Ana_Feat, fontsize=12)
plt.show()
#
# feature engineering: range analysis

range_whole = (10<train_fa[Ana_Feat]) & ( train_fa[Ana_Feat]<5000)
bad_count_range = train_tags[range_whole].value_counts()[1]
bad_percent_range = bad_count_range / float(sum(range_whole))
print 'bad_percent_range: '+str(bad_percent_range)+' bad:'+str(bad_count_range)+' both:' +str(sum(range_whole))


# whole range
#bad_count_whole = train_tags.value_counts()[1]
#both_count_whole = len(train_tags)
#bad_percent_whole = bad_count_whole / float(both_count_whole)
#print 'bad_percent_whole: '+str(bad_percent_whole)+' bad:'+str(bad_count_whole)+' both: '+str(both_count_whole)
