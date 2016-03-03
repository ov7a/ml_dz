#!/usr/bin/python
# -*- coding: utf-8 -*-

# Code source: Gael Varoquaux
#			  Andreas Muller
# Modified for documentation by Jaques Grobler
# forked by ov7a
# License: BSD 3 clause

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

from sklearn.metrics import roc_curve, auc, accuracy_score

import pandas
import sys
import itertools

if len(sys.argv) != 3:
	print "usage %s file outfile" % sys.argv[0]
	exit(1)

all_params = ['X9', 'X10', 'X13', 'X14', 'X15', 'X17', 'X18']

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
		 "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
		 "Quadratic Discriminant Analysis"]
classifiers = [
	KNeighborsClassifier(3),
	SVC(kernel="linear", C=0.025),
	SVC(gamma=2, C=1),
	DecisionTreeClassifier(max_depth=5),
	RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	AdaBoostClassifier(),
	GaussianNB(),
	LinearDiscriminantAnalysis(),
	QuadraticDiscriminantAnalysis()
]

ds = pandas.read_csv(sys.argv[1], ';').query("CLASS != 'U'")
with open(sys.argv[2], "w") as fout:
	for param1, param2 in itertools.combinations(all_params, 2):
		X = ds[[param1, param2]] 
		y = ds['CLASS'].apply(lambda x: 0 if x=='G' else 1)
		X = StandardScaler().fit_transform(X)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

		best_max_diff = (0, None, None)
		for name, clf in zip(names, classifiers):
			clf.fit(X_train, y_train)
			y_predicted = clf.predict(X_test)
			score = accuracy_score(y_predicted, y_test) 
		
			actual, predictions = y_test, y_predicted
			false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
			roc_auc = auc(false_positive_rate, true_positive_rate)
		
			diffs = true_positive_rate - false_positive_rate
		
			max_diff_index = diffs.argmax() 
			max_diff_value = diffs[max_diff_index]
			if (max_diff_value > best_max_diff[0]):
				best_max_diff = (max_diff_value, thresholds[max_diff_index], name)
		fout.write("Best diff for (%s, %s) is %0.2f with threshold %0.2f for %s classifier.\n" % (param1, param2, best_max_diff[0], best_max_diff[1], best_max_diff[2])) 	
