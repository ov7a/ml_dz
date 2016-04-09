import sys

if len(sys.argv) != 4:
	print "usage %s infile ensemble_config step" % sys.argv[0]
	exit(1)

from common import *

ensemble_classifiers = get_config(sys.argv[2])
step = int(sys.argv[3])


def fix_weights(ensemble_classifiers, individual_scores, scores):
	for ec, score in zip(ensemble_classifiers, individual_scores):
		diff = score[1] - scores[1]
		if (abs(diff) > 0.1):
			if (diff > 0):
				ec[-1] += 0.1
			elif ec[-1] >= 0.1:
				ec[-1] -= 0.1	

X_train_all, X_test_all, y_train, y_test = train_test_split(ds, y, test_size=.4)
for params, threshold, classifier, weight in ensemble_classifiers:
	X_train = X_train_all[params]
	classifier.fit(X_train, y_train)

all_scores = list()
all_weights = list()
for part in range(0, len(X_test_all), step):
	size = len(X_test_all[part:part+step])
	if not size:
		break
	result_prediction = np.zeros(size)
	weights_sum = 0
	individual_scores = list()
	for params, threshold, classifier, weight in ensemble_classifiers:
		X_test = X_test_all[params][part:part+size]
		y_predicted = classifier.predict_proba(X_test)[:,1] 
		y_predicted_binary = to_binary(y_predicted, threshold)
		scores = get_scores(y_predicted_binary, y_test[part:part+size])
		individual_scores.append(scores)
		result_prediction += y_predicted_binary*weight
		weights_sum += weight
	
	result_prediction = to_binary(result_prediction, weights_sum/2.0)
	scores = get_scores(result_prediction, y_test[part:part+size])
	#print "scores: P = %0.4f, R = %0.4f, F1 = %0.4f" % scores
	all_weights.append(map(lambda x: x[-1], ensemble_classifiers))
	fix_weights(ensemble_classifiers, individual_scores, scores)	
	all_scores.append(scores)

plt.plot(range(0, len(X_test_all), step), map(lambda x: x[0], all_scores), label = "F1_score")
plt.grid(True)
plt.legend(loc='lower left', shadow=True)
plt.show()	
for i in range(len(ensemble_classifiers)):
	plt.plot(range(0, len(X_test_all), step), map(lambda x: x[i], all_weights), label = "weight of " + str(i))
plt.grid(True)
plt.legend(loc='lower left', shadow=True)
plt.show()	
