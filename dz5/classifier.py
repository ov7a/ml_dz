import sys

if len(sys.argv) != 4:
	print "usage %s infile ensemble_config outfile" % sys.argv[0]
	exit(1)

from common import *

params_set = map(lambda x: x[0], get_config(sys.argv[2]))
train_sample = int(0.4*len(y))
X_train_all = ds[:train_sample]
X_test_all = ds[train_sample:]
y_train =y[:train_sample]
y_test = y[train_sample:]

n_estimators_variants = [3,5,10,20]
max_features_variants = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, "sqrt", "log2"]
min_samples_split_variants = [2,3,5]
min_samples_leaf_variants = [1,2,3,5]

bests = []
for params in params_set:
	print "Calculating for " + str(params)
	X_train = X_train_all[params]
	X_test = X_test_all[params]
	best_max_diff = (0, None, None, None, None)
	for max_features in max_features_variants:
		for min_samples_split in min_samples_split_variants:
			for min_samples_leaf in min_samples_leaf_variants:
				clf = DecisionTreeClassifier(max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
				clf.fit(X_train, y_train)		
				y_predicted = clf.predict_proba(X_test)[:,1] 	
				actual, predictions = y_test.as_matrix(), y_predicted
				
				false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
				diffs = (true_positive_rate - false_positive_rate)
				
				max_diff_index = diffs.argmax() 
				max_diff_value = diffs[max_diff_index]

				if (max_diff_value > best_max_diff[0]):
					best_max_diff = (max_diff_value, thresholds[max_diff_index], max_features, min_samples_split, min_samples_leaf)
		
	bests.append((params, best_max_diff))	
			
with open(sys.argv[3], "w") as fout:
	for params, best in bests:
		fout.write("Best for %s is diff %0.2f with threshold %0.2f for DecisionTreeClassifier(max_features=%s, min_samples_split=%s, min_samples_leaf=%s)\n" % ((str(params),) + best))	
