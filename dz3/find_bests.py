import sys

if len(sys.argv) != 3:
	print "usage %s file outfile" % sys.argv[0]
	exit(1)

from common import *

all_params = [	"Duration",
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
	"Dst_host_srv_serror_rate"
]
min_combo = 2
max_combo = 5

bests = []
for count in range(min_combo, max_combo+1):
	for params in itertools.combinations(all_params, count):
		print "Calculating for " + str(params)
		X = ds[list(params)] 
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
			
		bests.append((params, best_max_diff))	
			
bests.sort(key = lambda x: x[1][0])	
with open(sys.argv[2], "w") as fout:
	for params, best in bests:
		fout.write("Best diff for %s is %0.2f with threshold %0.2f for %s classifier.\n" % ((str(params),) + best))	
