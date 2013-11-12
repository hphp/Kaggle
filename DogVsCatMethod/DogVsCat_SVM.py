#!/usr/bin/python

'''
    2013-11-13
    written by hp_carrot
    Part 1:
    1.for each picture , resize . to 250*250 if bigger ,
        to 100*100 if smaller than 250 / or ignore smaller ones.
    2.cut to partions.
    3.check each partion if in [S_s,S_e]*[C_s,C_e]*[H_s,H_e]
    4.record in pickle/file

    Part2:
    1.get texture tiles info [index,3,patch_size,patch_size]
    2. record in pickle

    Part3:
    1.readin texture tiles info
    2.for each picture , calculate distance 
    3.normalization
    4.record in pickle.

    Part4:
    1.readin color_features/texture_features
    2.SVM training
    3.record W,b in pickles.

    Part5:
    0.readin texture tiles info , [W,b] ,
    1.for each test picture , resize , and patch up.
    2.extract features for each part. color features / texture features.
    3.warm up SVM using existing W,b
    4.test if true.
'''
