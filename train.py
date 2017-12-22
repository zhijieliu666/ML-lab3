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
from feature import *
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ensemble import AdaBoostClassifier
#这里是直接读取灰度图，灰度图在original文件夹里面
path1=[os.path.join('G:\\Users\\qqqqqq1997520\\Desktop\\original\\face\\face',f) for f in os.listdir('G:\\Users\\qqqqqq1997520\\Desktop\\original\\face\\face')]
path2 = [os.path.join('G:\\Users\\qqqqqq1997520\\Desktop\\original\\face\\nonface',f) for f in os.listdir('G:\\Users\\qqqqqq1997520\\Desktop\\original\\face\\nonface')]
ABC=AdaBoostClassifier(DecisionTreeClassifier(), 1)
im=[0 for i in range(1000)]
for i in range(500):
    im[i]=plt.imread(path1[i])
for i in arange(500,1000):
    im[i]=plt.imread(path2[(i%500)])
_feature=[0 for i in range(1000)]
for i in range(1000):
    feature=NPDFeature(im[i])
    _feature[i]=feature.extract()
    
feature_data=array(_feature)
y=[1 for i in range(1000)]
for i in range(500,1000):
    y[i]=-1;
y=array(y)
X_train, X_vali, y_train, y_vali = train_test_split(feature_data, y, test_size=0.3, random_state=37)

ABC=AdaBoostClassifier(DecisionTreeClassifier(),20)
ABC.fit(X_train,y_train)
predict=ABC.predict(X_vali,0)
classification_name=["Y","N"]
with open('report.txt', 'w') as f:
    f.write(classification_report(y_vali, predict, target_names=classification_name))
