#!/bin/sh

file_name='tmp_21000_0.78_50000'

grep "valid" $file_name | awk '{print $7}' > excel.valid
grep "train" $file_name | awk -F ',' '{if(NR%200==0)print $3}' | awk '{print $3}' > excel.train_cost
grep "train" $file_name | awk -F ',' '{if(NR%200==0)print $4}' | awk '{print $2}' > excel.train_err

gedit excel.valid
gedit excel.train_cost
gedit excel.train_err

