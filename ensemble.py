import numpy
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
import numpy as np
import pickle
from numpy import *
import matplotlib.image as mpimg
from skimage import io
import os
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import pickle

class AdaBoostClassifier:
    def __init__(self, weak_classifier, n_weakers_limit):
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.classifier_list = []
        self.alpha = zeros(n_weakers_limit)

    def fit(self,X,y):
        
        w=np.ones(X.shape[0])/(X.shape[0])
        for i in range(self.n_weakers_limit):
            classification=DecisionTreeClassifier(max_depth=1)
            classification.fit(X,y,sample_weight=w)
            self.classifier_list.append(classification)
            predict=classification.predict(X)
            x=np.sum(w[y!=predict])
            self.alpha[i]=0.5*np.log((1-x)/x)
            sumW=w*np.exp((-1)*y*predict)
            w=sumW/np.sum(sumW)
            
    def predict_scores(self, X):
        s=[]
        score=zeros(X.shape[0])
        for i in range(self.n_weakers_limit):
            s.append(self.classifier_list[i].predict(X))
            score += self.alpha[i] * s[i]
        return score

    def predict(self, X, threshold=0):
        score = self.predict_scores(X)
        score[score > threshold] = 1
        score[score < threshold] = -1
        return score

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
