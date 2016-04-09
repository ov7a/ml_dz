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

data_names = [
	"Duration",
	"Service",
	"Source_bytes",
	"Destination_bytes",
	"Count",
	"Same_srv_rate",
	"Serror_rate",
	"Srv_serror_rate",
	"Dst_host_count",
	"Dst_host_srv_count",
	"Dst_host_same_src_port_rate",
	"Dst_host_serror_rate",
	"Dst_host_srv_serror_rate",
	"Flag",
	
	"IDS_detection",
	"Malware_detection",
	"Ashula_detection",
	"Label",
	"Source_IP_Address",
	"Source_Port_Number",
	"Destination_IP_Address",
	"Destination_Port_Number",
	"Start_Time",
	"Duration2" ## same as Duration
]

#ds = pandas.read_csv(sys.argv[1], '\t', header=None, names = data_names, keep_default_na = False, na_values=[]).query("Label != '0'")
#y = ds['Label'].apply(lambda x: 1 if float(x)<0 else 0)
ds = pandas.read_csv(sys.argv[1], ';').query("CLASS != 'U'")
y = ds['CLASS'].apply(lambda x: 0 if x=='G' else 1)

to_binary = np.vectorize(lambda x,y: 1.0 if float(x)>=y else 0.0)

def get_config(infile):
	classifiers_dict = dict(zip(names, classifiers))
	with open(infile) as f:
		lines = filter(lambda l: len(l) and not l.startswith("#"), map(lambda l: l.strip('\n'), f.read().split("\n")))
		ensemble_classifiers = [[params.split(' '), float(threshold), classifiers_dict[name], 1.0] for params, threshold, name in map(lambda l: l.split('\t'), lines)]	
	return ensemble_classifiers
	
def get_scores(left, right):
	precision = precision_score(left, right)
	recall = recall_score(left, right)
	f1 = f1_score(left, right)
	return (precision, recall, f1)	
