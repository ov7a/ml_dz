import matplotlib.pyplot as plt

def get_recall(rates):
	sorted_by_rate = sorted(rates.values, key = lambda x: x[0], reverse = True)
	threshold = [1.0]
	TPR = [0.0]
	P = float(rates.p)
	for detected, actual in sorted_by_rate:
		threshold.append(detected)
		if (not actual):
			TPR.append(TPR[-1])
		else:
			TPR.append(TPR[-1] + 1.0/P)	
	return (threshold, TPR)
	
def get_precision(rates):
	sorted_by_rate = sorted(rates.values, key = lambda x: x[0], reverse = True)
	tp = 0.0
	fp = 0.0
	threshold = []
	result = []
	P = float(rates.p)
	for detected, actual in sorted_by_rate:
		threshold.append(detected)
		if (not actual):
			fp += 1
		else:
			tp += 1
		result.append(tp / (tp+fp))
	return (threshold, result)	
	
def plot(datas, typ):
	title = typ + "s of "
	for data in datas:
		data, hint = data
		threshold, result = data
		plt.plot(threshold, result, label = hint)
		title += " %s," % hint
	plt.axis([0, 1, 0, 1.05])
	plt.grid(True)
	plt.xlabel('Threshold')
	plt.ylabel(typ)
	plt.title(title[:-1]) 
	plt.legend(loc='lower left', shadow=True)
	plt.show()	
