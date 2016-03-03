import sys

if len(sys.argv) != 3:
	print "usage %s file outfile" % sys.argv[0]
	exit(1)

from common import *

all_params = ['X9', 'X10', 'X13', 'X14', 'X15', 'X17', 'X18']

bests = []
for param1, param2 in itertools.combinations(all_params, 2):
	X = ds[[param1, param2]] 
	X = StandardScaler().fit_transform(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

	best_max_diff = (0, None, None)
	for name, clf in zip(names, classifiers):
		clf.fit(X_train, y_train)		
		y_predicted = clf.predict_proba(X_test)[:,1] 	
	
		actual, predictions = y_test.as_matrix(), y_predicted
		false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
		#roc_auc = auc(false_positive_rate, true_positive_rate)
		
		diffs = (true_positive_rate - false_positive_rate)
	
		max_diff_index = diffs.argmax() 
		max_diff_value = diffs[max_diff_index]
		if (max_diff_value > best_max_diff[0]):
			best_max_diff = (max_diff_value, thresholds[max_diff_index], name)
			
	bests.append((param1, param2) + best_max_diff)			
	
bests.sort(key = lambda x: x[2])	
with open(sys.argv[2], "w") as fout:
	for best in bests:
		fout.write("Best diff for (%s, %s) is %0.2f with threshold %0.2f for %s classifier.\n" % best) 	
