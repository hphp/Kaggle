#!/usr/bin/python

import transform_data_to_format as tdtf

test_l1 = [1,2,3,4,5,6]
test_l2 = [1,2,3,4,5,6]
test_l = (test_l1,test_l2)
tdtf.write_content_to_csv(test_l,"../data/test.csv")
