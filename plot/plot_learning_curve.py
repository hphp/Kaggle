#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
"""
import time
import csv
import os
import datetime
import sys
import re
import copy
import traceback
import socket, fcntl, struct
import subprocess
from rpy2.robjects.packages import importr
from rpy2.robjects.lib import ggplot2
import rpy2.robjects as robjects

from ggplot import *
from pandas.core.api import Series, DataFrame, DateRange

#d=DataFrame({'a':[1,2,3,4,5,6,7,8,9,10],'t':[1,2,1,2,1,2,1,1,1,1],'b':[3,6,12,34,576,75,60,7,7,790]})


def read_and_monitor(file_list):
    n=0
    d={'idx':[],'src':[], 'val':[]}
    while True:
        line = None
        last=n
        for f in file_list:
            #print "read", f.name
            line = f.readline()
            line=line.strip()
            if line:
                n+=1
                d['idx'].append(n)
                d['src'].append(f.name)
                d['val'].append(float(line))
                print d
            #tmp = line.split('|')    
            #method_name=tmp[5][7:]    
        #if not line:
        #    time.sleep(0.1)
        if last==n:
            break
            
    d=DataFrame(d)    
    p=ggplot(aes(x='idx',y='val',color='factor(src)'),data=d)
    p+= geom_line()
    print(p)
    plt.show(1)


if __name__ == '__main__':
    data_file_list=[]
    if len(sys.argv)<2:
        print "usages: %s datafile1  [datafile2] [datafile3] ... " % sys.argv[0]
    for f in sys.argv[1:]:
        print "open ", f
        data_file_list.append(open(f,"r"))
    read_and_monitor(data_file_list)

                                
