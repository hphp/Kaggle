#!/usr/bin/python

'''
    noted by hp_carrot
    2013-11-01
    demo of the usage of PCA , decomposition.
'''

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import decomposition
import csv


def read_data(filname, limit=None):
    data = []
    labels = []

    csv_reader = csv.reader(open(filname, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index += 1
        if index == 1:
            continue

        labels.append(int(row[0]))
        row = row[1:]

        data.append(np.array(np.int64(row)))

        if limit != None and index == limit + 1:
            break

    return (data, labels)

print "Reading train data"
train, target = read_data("../data/train.csv",100)

print "PCA ing ... "
pca_components = [1, 2, 3, 4, 5, 10, 20, 25, 30, 50, 70, 100,400]
pca_fits = []

for comp_size in pca_components:
    #print "Fitting pca with %d components" % comp_size
    pca_fits.append(decomposition.PCA(n_components=comp_size).fit(train))
    #print "pca_fits size %d " % len(pca_fits)

figure = plt.figure()

t = np.array(target)

choosen_numbers = []

# this gives out the third element from argwhere t == x , counting from back.
choosen_numbers.append(np.argwhere(t == 0)[-3])
choosen_numbers.append(np.argwhere(t == 1)[-3])
choosen_numbers.append(np.argwhere(t == 2)[-3])
choosen_numbers.append(np.argwhere(t == 3)[-3])
choosen_numbers.append(np.argwhere(t == 4)[-3])
choosen_numbers.append(np.argwhere(t == 5)[-3])
choosen_numbers.append(np.argwhere(t == 6)[-3])
choosen_numbers.append(np.argwhere(t == 7)[-3])
choosen_numbers.append(np.argwhere(t == 8)[-3])
choosen_numbers.append(np.argwhere(t == 9)[-3])  

print "writing to pictures ... "
pca_index = 1
for n in choosen_numbers:
    for p in pca_fits:
        transformed = p.transform(train[n])
        #this is the 
        #print "Shape of transformed: "
        #print transformed.shape , transformed.shape[0] , transformed.shape[1]
        #print transformed
        reconstructed = p.inverse_transform(transformed)
        #print reconstructed.shape
        f = figure.add_subplot(10, len(pca_components), pca_index).imshow(np.reshape(reconstructed, (28, 28)), interpolation='nearest', cmap=cm.hot)  # cmap=cm.Greys_r)
        for xlabel_i in f.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        for xlabel_i in f.axes.get_yticklabels():
            xlabel_i.set_fontsize(0.0)
            xlabel_i.set_visible(False)
        for tick in f.axes.get_xticklines():
            tick.set_visible(False)
        for tick in f.axes.get_yticklines():
            tick.set_visible(False)
        pca_index += 1

plt.show()
