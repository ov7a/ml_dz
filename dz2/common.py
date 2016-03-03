import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score

import pandas
import sys
import itertools

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
		 "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
		 "Quadratic Discriminant Analysis"]
classifiers = [
	KNeighborsClassifier(3),
	SVC(kernel="linear", C=0.025, probability = True),
	SVC(gamma=2, C=1, probability = True),
	DecisionTreeClassifier(max_depth=5),
	RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	AdaBoostClassifier(),
	GaussianNB(),
	LinearDiscriminantAnalysis(),
	QuadraticDiscriminantAnalysis()
]

ds = pandas.read_csv(sys.argv[1], ';').query("CLASS != 'U'")
y = ds['CLASS'].apply(lambda x: 0 if x=='G' else 1)

to_binary = np.vectorize(lambda x,y: 1.0 if float(x)>=y else 0.0)
