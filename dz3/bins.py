import sys
import math

Gini_threshold = 0.2
alpha = 1.0

if len(sys.argv) != 4:
	print "usage %s infile ensemble_config outfile" % sys.argv[0]
	exit(1)

from common import *

ensemble = map(lambda x: x[0], get_config(sys.argv[2]))
X_train_all, X_test_all, y_train, y_test = train_test_split(ds, y, test_size=.4)

from sklearn.tree import DecisionTreeRegressor
def binningData(x, max_depth=5, min_samples_leaf=5):
	"""
	Suggest good threshold to bin data.   
	kittipat@
	Jul 1, 2015
	"""
	X = np.swapaxes(np.array([x]),0,1) 
	y = x
	clf_1 = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
	clf_1.fit(X, y)
	y_hat = clf_1.predict(X)

	thrs_out = np.unique( clf_1.tree_.threshold[clf_1.tree_.feature > -2] )
	thrs_out = np.sort(thrs_out)
	with_edges = np.insert(thrs_out, [0,0,len(thrs_out),len(thrs_out)], [float("-inf"), x.min(), x.max(), float("+inf")])
	return with_edges

def reverse_logistic(x):
	return -math.log(1.0/x-1.0)/alpha

def get_bin_index(value, arr): #FIXME: optimize
	for i in range(len(arr)-1):
		if (value >= arr[i]) and (value < arr[i+1]):
			return i

def get_bin_indices(row, tresholds):
	return tuple([get_bin_index(value, arr) for value, arr in zip(row, tresholds)])
	
def get_bins(X_train, y_train):
	matrix = X_train.as_matrix()
	ground_truth = y_train.as_matrix()
	dimensions = X_train.shape[1]
	tresholds = [binningData(matrix[:,d]) for d in range(dimensions)]
	shape = map(len,tresholds)
	hits = np.zeros(shape)
	norm = np.zeros(shape)
	for i in range(len(matrix)):
		row = matrix[i]
		y = ground_truth[i]
		indices = get_bin_indices(row, tresholds)
		if y == 0:
			norm[indices] += 1
		else:
			hits[indices] += 1	
	it = np.nditer(hits, flags=['multi_index'])
	probs = np.full(shape, 0.5)
	while not it.finished:
		indices = it.multi_index
		it.iternext()
		total = float(hits[indices] + norm[indices])
		if total == 0:
			continue
		dh = float(hits[indices])/total
		dn = float(norm[indices])/total
		Gini = 1 - dh ** 2 - dn ** 2
		if Gini < Gini_threshold:
			probs[indices] = dh
	result = {
		'tresholds': tresholds,
		'values': probs
	}
	return result


def predict(X_test_row, bins):
	indices = get_bin_indices(X_test_row, bins['tresholds'])
	p = bins['values'][indices]
	return p
	
result_prediction = np.zeros(len(X_test_all))

for params in ensemble:
	X_train = X_train_all[params]
	X_test = X_test_all[params]
	bins = get_bins(X_train, y_train)
	y_predicted = np.apply_along_axis(predict, 1, X_test, bins)
	result_prediction += y_predicted
	
result_prediction = to_binary(result_prediction, len(ensemble)/2.0)

with open(sys.argv[3], "w") as f:
	scores = get_scores(result_prediction, y_test)
	f.write("Ensemble scores: P = %0.4f, R = %0.4f, F1 = %0.4f\n" % scores)
