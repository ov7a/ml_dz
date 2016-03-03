from utils.Rates import write_rates
import sys

def find_threshold(rates, checker, increasing = False, accuracy = 0.0001):
	def find(low, hi):
		current = (low + hi)/2.0
		if (hi - low < accuracy):
			return hi if increasing else low
	
		rates.calculate(current)
		#write_rates(sys.stdout, rates, current, "ololo")
	
		result = checker(rates)
		#print "calculated for", (low , hi, current, result), "\n"
		if (not result and increasing) or (result and not increasing):
			return find(current, hi)
		else:
			return find(low, current)
			
	return find(0.0, 1.0)		
