import matplotlib.pyplot as plt

def get(rates):
	sorted_by_rate = sorted(rates.values, key = lambda x: x[0], reverse = True)
	AUC = 0.0
	FPR = [0.0]
	TPR = [0.0]
	P = float(rates.p)
	N = float(rates.n)
	for detected, actual in sorted_by_rate:
		if (not actual):
			FPR.append(FPR[-1] + 1.0/N)
			TPR.append(TPR[-1])
			AUC = AUC + 1.0*TPR[-1]/N
		else:
			FPR.append(FPR[-1])	
			TPR.append(TPR[-1] + 1.0/P)	
	Gini = 2*AUC - 1
	return (FPR, TPR, Gini)

		
def plot(datas):
	plt.plot([0, 1], [0, 1])
	title = ""
	for data in datas:
		data, hint = data
		FPR, TPR, Gini = data
		plt.plot(FPR, TPR, label = hint)
		title += "%s (Gini = %f)\n" % (hint, Gini)
	plt.axis([0, 1, 0, 1])
	plt.grid(True)
	plt.xlabel('FP rate')
	plt.ylabel('TP rate')
	plt.title(title[:-1]) 
	plt.legend(loc='lower right', shadow=True)
	plt.show()
