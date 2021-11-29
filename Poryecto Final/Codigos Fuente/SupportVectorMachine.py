from scipy.stats import mode
import numpy as np
#from mnist import MNIST
from time import time
import pandas as pd
import os
import matplotlib.pyplot as matplot
import matplotlib
%matplotlib inline

import random
matplot.rcdefaults()
from IPython.display import display, HTML
from itertools import chain
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sb
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC, LinearSVC
import warnings
warnings.filterwarnings('ignore')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/')

train = mnist.train.images
validation = mnist.validation.images
test = mnist.test.images

trlab = mnist.train.labels
vallab = mnist.validation.labels
tslab = mnist.test.labels

train = np.concatenate((train, validation), axis=0)
trlab = np.concatenate((trlab, vallab), axis=0)

svm = LinearSVC(dual=False)
svm.fit(train, trlab)

svm.coef_
svm.intercept_
pred = svm.predict(test)

accuracy_score(tslab, pred) # Accuracy


cm = confusion_matrix(tslab, pred)
matplot.subplots(figsize=(10, 6))
sb.heatmap(cm, annot = True, fmt = 'g')
matplot.xlabel("Predicted")
matplot.ylabel("Actual")
matplot.title("Confusion Matrix")
matplot.show()


