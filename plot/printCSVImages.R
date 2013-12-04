source("./myImagePlot.R")
library("matrixStats")

images=read.csv("/Users/xcbfreedom/projects/data/Kaggle/DogVsCatData/DogVsCat_head_train_feature_2500.csv",nrows=1500) 
lable=1
w=50
h=50
#images_for_special_label=colMeans(subset(images,images[,1]==lable))
images_for_special_label=colMins(subset(images,images[,1]==lable))

#images_for_special_label=colMaxs(subset(images,images[,1]==lable))

images_for_special_label=colMedians(subset(images,images[,1]==lable)) 

m=rowToMatrix(images_for_special_label[2:2501],50,50)

#display the image 
#displayMatrix(m) 

plotMatrix(resize(m,28,28)) 