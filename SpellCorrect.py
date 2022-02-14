# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 20:52:49 2018

@author: Qing Gong
"""
from Levenshtein import *


#toxiclist = ['fuck']
#
#s = 'ho fuckkkkkkk! d_e i&^d f**k uyn HO'
# wish to return 'ho, fuck! i&^d fuck u'

#s_c = s.split()[:]

def word_freq(string_list):
    ''' count the word frequency in the input string. return a dictionary:
    key: word, value:count '''
    word_dict = {}
    for s in string_list:
        ss = s.lower().split()      
        for w in ss:
            if w in word_dict:
                word_dict[w] += 1
            else:
                word_dict[w] = 1
    return word_dict



class ListCorrector(object):
    def __init__(self, string_list, word_dict):
        self.string_list = string_list
        self.word_dict = word_dict
        
    def spell_correct(self, input_word, thres_fq, target_word_list):
        ''' If the input word are close to target_word,
        then replace it with target_word and return the revised string.'''
        maxr_i, maxr = 0, 0.49
        for i, target_word in enumerate(target_word_list):
            if self.word_dict[input_word] < thres_fq:
                if target_word in input_word:
                    return target_word
                if ratio(target_word, input_word) > maxr:
                    maxr = ratio(target_word, input_word)
                    maxr_i = i
        
        if maxr >= 0.6:
            return target_word_list[maxr_i]
        
        return input_word
    
    def list_correct(self, thres_fq, target_word_list):
        '''correct each string in the input list, and return the corrected list'''
        self.corrected = ['']*len(self.string_list)
        for i, s in enumerate(self.string_list):
            ss = s.lower().split()
            self.corrected[i] = ' '.join([self.spell_correct(w, thres_fq, target_word_list) for w in ss])
        return self.corrected

#list_correct([s],toxiclist)
#Out[61]: ['ho fuck d_e i&^d fuck uyn ho']


#for i,word in enumerate(s_c):
#    print word
#    if frequency_of_word < tol:
#        if (toxiclist[0] in word) or (ratio(toxiclist[0], word)>=0.5):
#             s_c[i] = toxiclist[0]

#count = CountVectorizer(strip_accents='unicode', analyzer='word')
#s_doc = count.fit_transform([s])
#
#count.get_feature_names()
#s_doc.toarray()
#
#word_freq(s)
