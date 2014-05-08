from sklearn.linear_model import Perceptron
import csv
from datetime import date
from collections import defaultdict as dt
import preprocess as pp
from copy import deepcopy as dc
import heapq

# read csv, format date, save to 'data'
csvfile = file('t_alibaba_data.csv','rb')
reader = csv.reader(csvfile)
data = []
for line in reader:
    if reader.line_num == 1:
        continue
    line[3] = date(*pp.parse_date(line[3]))
    line[2] = int(line[2])
    data.append(line)
csvfile.close()

# spliting data to 4 parts
mon4,mon5,mon6,mon7 = pp.split_data(data)

# training set: train_d aka x, (mon4, mon5); train_t aka y, (mon6)
# testing set: test_d aka x, (mon4, mon5, mon6); test_t aka y, (mon7)
train_d = dc(mon4)
train_d.extend(mon5)
test_d = dc(train_d)
test_d.extend(mon6)
train_t = dc(mon6)
test_t = dc(mon7)

# processing data
train_d = pp.process_activity(train_d)
train_t = pp.process_activity(mon6)
train_d = pp.normalization(train_d)

train_x,train_y = pp.get_train_data(train_d,train_t)
test_d = pp.process_activity(test_d)
test_d = pp.normalization(test_d)
test_t = pp.process_activity(test_t)

# using percetron to train model
pcpt = Perceptron()
pcpt.fit(train_x, train_y)

# geting 3000 best prediction data
result = heapq.nlargest(2000,test_d,lambda x:pcpt.decision_function(test_d[x]))

# calculating the quality of result
precision, recall, f1 = pp.get_comments(result, test_t)
print "Precision rate: %f\nRecall rate: %f\nF1: %f\n" % (precision,recall,f1)
