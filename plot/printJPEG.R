#!/usr/bin/R
source("./myImagePlot.R")
library("EBImage")
image=readImage("/Users/xcbfreedom/projects/data/Kaggle/DogVsCatData/train/cat.13.jpg")


#display the image 
displayMatrix(image*255) 

#display the one color component
r=getFrame(image,1)
plotMatrix(resize(r*255,28,28)) 
