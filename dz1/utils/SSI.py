from utils.Rates import *
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt

def get_date(record):
	date_str = record["EVENT_TIME"].strip('"')
	date_tz = float(record["X22"]) if record["X22"] != "NULL" else 0.0
	date = datetime.strptime(date_str,'%Y-%m-%d %H:%M:%S') - timedelta(hours = date_tz) # FFS, store dates in unix ts, please
	return date

def plot_SSI(datasets, hint, dateformat = '%Y-%m-%d'):
	title = "SSI of " + hint
	for dataset in datasets:
		data, hint = dataset
		dates, ssi = data
		plt.plot(range(len(dates)), ssi, label = hint)
	#title += " %s," % hint
	
	plt.grid(True)
	plt.xlabel('Dates')
	plt.xticks(range(len(dates)), map(lambda x: x.strftime(dateformat), dates), rotation = 60)
	plt.ylabel('SSI')
	plt.title(title) 
	plt.legend(loc='lower left', shadow=True)
	plt.show()	

class SSIReader(Reader):
	def __init__(self, getter, class_getter, date_from, date_to, date_delta = timedelta(hours = 1)):
		self.date_from = date_from
		self.date_fo = date_to
		self.date_delta = date_delta
		self.date_readers = list()
		current = date_from
		while current < date_to + date_delta: #TODO: make normal ds
			self.date_readers.append((current, RatesReader(getter, class_getter))) #last is not used
			current += date_delta	
		
	def get_reader(self, date):
		if date < self.date_readers[0][0]:
			return None
		for i in range(len(self.date_readers)-1): #TODO bin search
			if date >= self.date_readers[i][0] and date < self.date_readers[i+1][0]:
				return self.date_readers[i][1]
		return None
		
	def read(self, record):	
		date = get_date(record)
		reader = self.get_reader(date)
		if reader != None:
			reader.read(record)
			
	def get_data(self, func, threshold_points=100):		
		ssi_by_date = list()
		dates = list()
		for i in range(len(self.date_readers)-2): #TODO: optimize
			rates1 = self.date_readers[i][1].rates
			rates2 = self.date_readers[i+1][1].rates
			ssi = 0.0
			for x in range(threshold_points):
				point = float(x)/float(threshold_points)
				rates1.calculate(point)
				rates2.calculate(point)
				f1 = float(func(rates1))
				f2 = float(func(rates2))
				if f1 == f2 or f1 == 0 or f2 ==0: #FIXME
					continue
				ssi += (f1-f2)*math.log(f1/f2)
			ssi_by_date.append(ssi)	
			dates.append(self.date_readers[i+1][0])
		#print dates, ssi_by_date	
		return (dates, ssi_by_date)
