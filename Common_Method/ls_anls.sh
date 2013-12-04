#!/bin/sh

grep "train_cost" $1 | awk '{if((NR-1)%100==0)print $6,"\011",$9,"\011",$13,"\011",$17}'
