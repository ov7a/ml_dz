import sys

if len(sys.argv) != 4:
	print "usage %s infile ensemble_config outfile" % sys.argv[0]
	exit(1)

from common import *

ensemble_classifiers = get_config(sys.argv[2])

X_train_all, X_test_all, y_train, y_test = train_test_split(ds, y, test_size=.4)
result_prediction = np.zeros(len(X_test_all))

for params, threshold, classifier in ensemble_classifiers:
	X_train = X_train_all[params]
	X_test = X_test_all[params]
	classifier.fit(X_train, y_train)
	y_predicted = classifier.predict_proba(X_test)[:,1] 
	y_predicted_binary = to_binary(y_predicted, threshold)
	result_prediction += y_predicted_binary
	
result_prediction = to_binary(result_prediction, len(ensemble_classifiers)/2.0)

with open(sys.argv[3], "w") as f:
	scores = get_scores(result_prediction, y_test)
	f.write("Ensemble scores: P = %0.4f, R = %0.4f, F1 = %0.4f\n" % scores)
	
	
