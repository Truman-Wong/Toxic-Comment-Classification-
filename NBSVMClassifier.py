# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 10:46:39 2018

@author: Qing Gong
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression

class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1, C=1.0, dual=False, n_jobs=1):
        self.alpha = alpha
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y): # smoothed count vectors/ likelihood
            p = x[y==y_i].sum(0)
            return (p+self.alpha) / ((y==y_i).sum()+self.alpha)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))# log(p1/p2) -- for elementwise multiplication
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self