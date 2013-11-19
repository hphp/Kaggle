#!/usr/bin/R
source("./myImagePlot.R")
library("EBImage")
#lenac = readImage(system.file("images", "lena-color.png", package="EBImage"))
image=readImage("../../data/dogs_vs_cats/cat.13.jpg")
r=getFrame(image,2)
#b=as.matrix(images[6,2:785],rownames.force=NA) 
r=resize(r,28,28)
b=ceiling(r*256)
#display the image 
#print(b)
myImagePlot(t(imageData(b))) 
