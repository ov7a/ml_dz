#!/usr/bin/python
# -*- coding: utf-8 -*-

# Code source: Gael Varoquaux
#              Andreas Muller
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

if len(sys.argv) != 4:
    print "usage %s file param1 param2" % sys.argv[0]
    exit(1)

infile = sys.argv[1]
param1 = sys.argv[2]
param2 = sys.argv[3]

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant\nAnalysis",
         "Quadratic Discriminant\nAnalysis"]
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

data = pandas.read_csv(infile, ';')
filtered = data.query("CLASS != 'U'")

datasets = [filtered]

to_binary = np.vectorize(lambda x: 1.0 if x>=0.5 else 0.0)

figure = plt.figure(figsize=(27, 9)) # w,h tuple in inches SUCH SCIENTIFIC
plotsNX = len(classifiers) + 1
plotsNY = len(datasets) + 1
i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X = ds[[param1, param2]] 
    y = ds['CLASS'].apply(lambda x: 0 if x=='G' else 1)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(plotsNY, plotsNX, i)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(plotsNY, plotsNX, i)
        clf.fit(X_train, y_train)
        y_predicted = clf.predict_proba(X_test)[:,1]     
        y_predicted_binary = to_binary(y_predicted)
        score = accuracy_score(y_predicted_binary, y_test) 

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1
        
        ax = plt.subplot(plotsNY, plotsNX, plotsNX + i - 1)
        actual, predictions = y_test, y_predicted
        false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        ax.plot(false_positive_rate, true_positive_rate)
        ax.plot([0,1], [0,1])
        ax.grid(True) 
        max_diff = (true_positive_rate - false_positive_rate).max() 
        ax.set_title('AUC = %0.2f\nGini = %0.2f\nmax_diff = %0.2f' % (roc_auc, (roc_auc * 2) - 1, max_diff))   

figure.subplots_adjust(left=.02, right=.98)
plt.show()
