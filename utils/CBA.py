from utils.Rates import RatesReader
import matplotlib.pyplot as plt

def saved(rates, amounts, threshold):
	return sum(amount for detect_and_actual, amount in zip(rates.values, amounts) if detect_and_actual[1] and detect_and_actual[0] >= threshold)

def loosed(rates, amounts, threshold):
	missed = sum(amount for detect_and_actual, amount in zip(rates.values, amounts) if detect_and_actual[1] and detect_and_actual[0] < threshold)
	call_rate = 0.9
	call_cost = 40
	false_detect_charge  = 3000
	calls = (rates.tp + rates.fp)*call_rate*call_cost
	return calls + rates.fp * call_rate * false_detect_charge #+ missed

def checker(rates, amounts, threshold):
	amount_saved = saved(rates, amounts, threshold)
	amount_loosed = loosed(rates, amounts, threshold)
	#print threshold, amount_saved, amount_loosed, amount_saved-amount_loosed
	return amount_saved - amount_loosed

def get_amount(record):
	if not len(record["AMOUNT"]):
		return None
	amount = float(record["AMOUNT"])	
	currency = record["X12"]
	if currency == "RUB":
		return amount
	elif currency == "USD":	
		return amount*75.4
	elif currency == "EUR":	
		return amount*83.1
	elif currency == "XAU":	
		return amount*94292.35
	elif currency == "XAG":	
		return amount*1183
	else:
		print "Unknown currency: ", currency

def find_cba_threshold(rates, amounts, accuracy = 0.0001):
	increasing = False
	def find(low, hi):
		current = (low + hi)/2.0
		if (hi - low < accuracy):
			return hi if increasing else low
	
		rates.calculate(current)		
		result = checker(rates, amounts, current) > 0
		if (not result and increasing) or (result and not increasing):
			return find(current, hi)
		else:
			return find(low, current)
			
	result = find(0.0, 1.0)		
	balanse = checker(rates, amounts, result)
	return (result, balanse)

def plot_cba(reader, hint): #TODO: refactor, optimize
	points = 1000
	threshold = map(lambda x: x/float(points), range(points))
	amounts = reader.amounts
	rates = reader.rates
	amount_saved = map(lambda t: saved(rates, amounts, t), threshold)
	amount_loosed = map(lambda t: loosed(rates, amounts, t), threshold)
	diff = [x - y for x, y in zip(amount_saved, amount_loosed)]
	#plt.plot(threshold, amount_saved, label = 'saved')
	#plt.plot(threshold, amount_loosed, label = 'loosed')
	plt.plot(threshold, diff, label = 'diff')
	plt.grid(True)
	plt.xlabel('threshold')
	plt.ylabel('amount')
	plt.title(hint + " analysis") 
	plt.legend(loc='lower right', shadow=True)
	plt.show()

class CBAReader(RatesReader):
	def __init__(self, getter, class_getter):
		RatesReader.__init__(self, getter, class_getter)
		self.get_amount = get_amount
		self.amounts = []
		
	def read(self, record):
		amount = self.get_amount(record)
		if not amount:
			return
		super(CBAReader, self).read(record)
		self.amounts.append(amount) 
