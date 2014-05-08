# Pre-process the csv data to training data
import csv
from datetime import date
from collections import defaultdict as dt

def parse_date(d):
    '''format to python date'''
    dd = d.decode('gbk')
    month = int(dd[0])
    if len(dd) == 5:
        day = 10 * int(dd[2]) + int(dd[3])
    else:
        day = int(dd[2])
    return 2013, month, day

def split_data(dd):
    '''split data to 4 parts'''
    m4 = filter(lambda x:x[3]<=date(2013,5,15),dd)
    m5 = filter(lambda x:date(2013,5,15)<x[3]<=date(2013,6,15),dd)
    m6 = filter(lambda x:date(2013,6,15)<x[3]<=date(2013,7,15),dd)
    m7 = filter(lambda x:date(2013,7,15)<x[3]<=date(2013,8,15),dd)
    return m4,m5,m6,m7

def process_activity(ddd):
    '''process data to format like {(uid,bid):[0,1,2,3]}'''
    cc = dt(int)
    dddd = [(row[0],row[1],row[2]) for  row in ddd]
    for k in dddd:
        cc[k] += 1
#    return cc.items()
    tmp = {}
    for it in list(cc.items()):
        tmp[(it[0][0],it[0][1])] = [0,0,0,0]
    for it in list(cc.items()):
        tmp[(it[0][0],it[0][1])][it[0][2]] = it[1]
    return tmp #{(1,2):[1,2,3,4]}


def process_features(tmp,n):
    '''weighted sum'''
    data = {}
    for k in tmp:
        data[k] = tmp[k][0]*.05 + tmp[k][1]*.5 + tmp[k][2]*.15 + tmp[k][3]*.3
        data[k] /= n
    return data


def get_train_data(d1,d2):
    '''process data for training'''
    out_x,out_y = [],[]
    for k in d1:
        if (k in d2) and d2[k][1]:
            out_y.append(1)
        else:
            out_y.append(-1)
        out_x.append(d1[k])
    return out_x,out_y

def print_data_10(data):
    '''print 10 of the data set for intuitive checking'''
    count = 0
    for k in data:
        count += 1
        print k,data[k]
        if count == 20:
            return

def normalization(train_d):
    '''normalize the features'''
    max_num = []
    for k in range(4):
        tmp = max(train_d, key=(lambda x:train_d[x][k]))
        max_num.append(train_d[tmp][k])
    for key in train_d:
        for k in range(4):
            train_d[key][k] = float(train_d[key][k]) / max_num[k]
    return train_d

def get_comments(res,test):
    '''get precision, recall and f1'''
    hits = 0
    for k in res:
        if (k in test.keys()) and test[k][1]:
            hits += 1
    pre = float(hits)/len(res)
    overall = 0
    for k in test:
        if test[k][1]:
            overall += 1
    rec = float(hits)/overall
    f1 = (2 * pre * rec)/(pre + rec)
    return pre,rec,f1
