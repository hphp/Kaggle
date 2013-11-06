#!/usr/bin/python

import os

piclist = os.listdir("train/")
t_range = len(piclist)
period = 1000
total = int(t_range/period)
print total
for rr in range(total):
    start = rr * 1000
    end = min((rr+1)*1000,t_range)
    cmd = "python feature_extraction.py " + str(start) +  " " + str(end)
    os.system(cmd)
