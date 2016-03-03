import sys
from datetime import datetime, timedelta

from utils.Rates import *
from utils import ROC, PR
from utils.CBA import CBAReader, find_cba_threshold, plot_cba
from utils.Threshold import *
from utils.SSI import SSIReader, plot_SSI


def load(fname, readers):
	with open(fname) as f:
		headers = f.readline().strip("\r\n").split(";")
		for line in f:
			if len(line) < 2:
				continue
			splitted = line.strip("\r\n").split(";")
			record = dict(zip(headers, splitted))
			for reader in readers.itervalues():
				reader.read(record)
				
def class_getter(record):
	try:
		return record["CLASS"]
	except:
		print record	

def vote(vars, thresholds):
	count = 0
	for (var, threshold) in zip(vars, thresholds):
		if var >= threshold:
			count +=1
	if count > len(vars)/2:
		return 1.0
	else:
		return 0.0			
	

if __name__=="__main__":
	if len(sys.argv) != 3:
		print "usage: %s in_filename out_filename" % sys.argv[0]
		exit(1)
	
	readers = dict()
	
	names = ["p%d_Fraud" % (i+1) for i in range(3)]
	
	
	for name in names: 
		readers[name] = RatesReader(lambda x, name=name: x[name], class_getter)
		readers["cba_" + name] = CBAReader(lambda x, name=name: x[name], class_getter)
	
	readers["ensemble3"] = RatesReader(lambda x: (float(x["p1_Fraud"]) + float(x["p2_Fraud"]) + float(x["p3_Fraud"]))/3.0, class_getter)	
	names.append("ensemble3")
	
	readers["ensemble0.5"] = RatesReader(lambda x: vote(map(float,[x["p1_Fraud"], x["p2_Fraud"], x["p3_Fraud"]]), [0.5, 0.5, 0.5]), class_getter)	
	readers["ensemble0.8"] = RatesReader(lambda x: vote(map(float,[x["p1_Fraud"], x["p2_Fraud"], x["p3_Fraud"]]), [0.8, 0.8, 0.8]), class_getter)	
	readers["ensemble0.700"] = RatesReader(lambda x: vote(map(float,[x["p1_Fraud"], x["p2_Fraud"], x["p3_Fraud"]]), [0.700, 0.700, 0.700]), class_getter)	#yep, manually
	readers["ensemble0.701"] = RatesReader(lambda x: vote(map(float,[x["p1_Fraud"], x["p2_Fraud"], x["p3_Fraud"]]), [0.701, 0.701, 0.701]), class_getter)	
	names2 = ["ensemble0.5", "ensemble0.8", "ensemble0.700", "ensemble0.701"]
	
	for name in names[:-1]:
		readers[name+"_SSI_by_weeks"] = SSIReader(lambda x, name=name: x[name], class_getter, datetime(2015,12,15), datetime(2016,01,12), timedelta(weeks=1))
		readers[name+"_SSI_by_4days"] = SSIReader(lambda x, name=name: x[name], class_getter, datetime(2015,12,15), datetime(2016,01,12), timedelta(days=4))
		readers[name+"_SSI_by_2days"] = SSIReader(lambda x, name=name: x[name], class_getter, datetime(2015,12,15), datetime(2016,01,12), timedelta(days=2))
		readers[name+"_SSI_by_day"] = SSIReader(lambda x, name=name: x[name], class_getter, datetime(2015,12,15), datetime(2016,01,12), timedelta(days=1))
				
	load(sys.argv[1], readers)
	
	with open(sys.argv[2], "w") as f:
		#Tasks 1-4 and 6
		
#		for name in names: 
#			rates = readers[name].rates
#			write_rates(f, rates, 0.6875, name)
#			threshold = find_threshold(rates, lambda x: x.fp_rate <= 0.2, True)
#			f.write("Threshold for %s when fp rate <=0.2 is %f\n" % (name, threshold))
#			f.write("\n")

#		ROC.plot([(ROC.get(readers[name].rates), name) for name in names])
#		
#		#Task 5
#		
#		for name in names2:
#			rates = readers[name].rates
#			write_rates(f, rates, 0.5, name) 
#		f.write("Threshold for vote ensemble when fp rate is zero is between 0.700 and 0.701\n")

		#Task 7
		#skipped
		
		#Task 8
		#PR.plot([(PR.get_recall(readers[name].rates), name) for name in names], "Recall")
		#PR.plot([(PR.get_precision(readers[name].rates), name) for name in names], "Precision")
		
#		recall = lambda rates: rates.tp_rate
#		precision = lambda rates: rates.tp / (rates.tp + rates.fp) if (rates.tp + rates.fp) else 0
#		f1 =  lambda rates: 2.0 * rates.tp / (rates.p + rates.tp + rates.fp)
#		
#		plot_SSI([(readers[name+"_SSI_by_weeks"].get_data(recall), name) for name in names[:-1]], "recall by weeks")
#		#plot_SSI([(readers[name+"_SSI_by_4days"].get_data(recall), name) for name in names[:-1]], "recall by 4 days")
#		#plot_SSI([(readers[name+"_SSI_by_2days"].get_data(recall), name) for name in names[:-1]], "recall by 2 days")
#		plot_SSI([(readers[name+"_SSI_by_day"].get_data(recall), name) for name in names[:-1]], "recall by day")
#		
#		#plot_SSI([(readers[name+"_SSI_by_weeks"].get_data(precision), name) for name in names[:-1]], "precision by weeks")
#		#plot_SSI([(readers[name+"_SSI_by_4days"].get_data(precision), name) for name in names[:-1]], "precision by 4 days")
#		#plot_SSI([(readers[name+"_SSI_by_2days"].get_data(precision), name) for name in names[:-1]], "precision by 2 days")
#		#plot_SSI([(readers[name+"_SSI_by_day"].get_data(precision), name) for name in names[:-1]], "precision by day")
#		
#		plot_SSI([(readers[name+"_SSI_by_weeks"].get_data(f1), name) for name in names[:-1]], "f1 score by weeks")
#		#plot_SSI([(readers[name+"_SSI_by_4days"].get_data(f1), name) for name in names[:-1]], "f1 score by 4 days")
#		#plot_SSI([(readers[name+"_SSI_by_2days"].get_data(f1), name) for name in names[:-1]], "f1 score by 2 days")
#		plot_SSI([(readers[name+"_SSI_by_day"].get_data(f1), name) for name in names[:-1]], "f1 score by day")
#				
#		#Task 9
#		f.write("Currency is X12, timezone is X22, platform is X11 + X16, IP is X23\n")
		
		#Task 10
		cba_name = "cba_p2_Fraud"
		(cba_threshold, balanse) = find_cba_threshold(readers[cba_name].rates, readers[cba_name].amounts)
		f.write("Threshold for %s is %f with balance %f\n" % (cba_name, cba_threshold, balanse))
		plot_cba(readers[cba_name], cba_name)
		
