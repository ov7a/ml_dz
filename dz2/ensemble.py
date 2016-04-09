import sys

if len(sys.argv) != 4:
	print "usage %s infile ensemble_config outfile" % sys.argv[0]
	exit(1)

from common import *

classifiers_dict = dict(zip(names, classifiers))
with open(sys.argv[2]) as f:
	lines = filter(lambda l: len(l) and not l.startswith("#"), map(lambda l: l.strip('\n'), f.read().split("\n")))
	ensemble_classifiers = [(params.split(' '), float(threshold), classifiers_dict[name]) for params, threshold, name in map(lambda l: l.split('\t'), lines)]	

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

def get_scores(left, right):
	precision = precision_score(left, right)
	recall = recall_score(left, right)
	f1 = f1_score(left, right)
	return (precision, recall, f1)

with open(sys.argv[3], "w") as f:
	scores = get_scores(result_prediction, y_test)
	f.write("Ensemble scores: P = %0.2f, R = %0.2f, F1 = %0.2f\n" % scores)
	for i in range(1,6):
		result = to_binary(X_test_all[["p%d_Fraud" % i]], 0.5)
		scores = get_scores(result, y_test)
		f.write("p%d_Fraud scores (0.5 threshold): P = %0.2f, R = %0.2f, F1 = %0.2f\n" % ((i,) + scores))
	
	
