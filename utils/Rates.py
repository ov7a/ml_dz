from utils.Reader import Reader

def is_fraud(char):
	if char == 'F':
		return True
	elif char == 'G':
		return False
	elif char == 'U':	
		return None	

def write_rates(f, rates, threshold, hint):
	rates.calculate(threshold)
		
	f.write("Rates for %s with threshold %f:\n" % (hint, threshold))
	f.write("\ttp rate: %f\n" % rates.tp_rate)
	f.write("\tfp rate: %f\n" % rates.fp_rate)
	f.write("\ttn rate: %f\n" % rates.tn_rate)
	f.write("\tfn rate: %f\n" % rates.fn_rate)

class Rates(object):
	def __init__(self, values = None):
		if values:
			self.values = values
		else:
			self.values = list()
		self.p = 0	
		self.n = 0
	
	def add(self, detected, actual):
		if actual:
			self.p += 1 
		else:
			self.n += 1	
		self.values.append((detected, actual))
	
	def calculate(self, threshold = 0.5):
		self.tp = 0
		self.fp = 0
		self.tn = 0
		self.fn = 0
		for detected, actual in self.values:
			if actual and detected >= threshold:
				self.tp += 1
			elif actual and detected < threshold: 
				self.fn += 1
			elif not actual and detected >= threshold:
				self.fp += 1
			else:
				self.tn += 1	
			
		if self.n:	
			self.fp_rate = float(self.fp) / self.n
			self.tn_rate = float(self.tn) / self.n		
		else:
			self.fp_rate = float('nan')	
			self.tn_rate = float('nan')	
		
		if self.p:
			self.tp_rate = float(self.tp) / self.p
			self.fn_rate = float(self.fn) / self.p
		else:
			self.tp_rate = float('nan')
			self.fn_rate = float('nan')	
		
class RatesReader(Reader):
	def __init__(self, getter, class_getter):
		self.getter = getter
		self.class_getter = class_getter
		self.rates = Rates()
		
	def read(self, record):
		got = float(self.getter(record))
		expected = self.class_getter(record)
		fraud = is_fraud(expected)
		if fraud is None:
			return	
		self.rates.add(got, fraud)		
