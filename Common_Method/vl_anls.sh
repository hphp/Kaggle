#!/bin/sh

grep "validation error" $1 | awk '{if((NR-1)%40==0)print $7}'
