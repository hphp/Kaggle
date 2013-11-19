source("./myImagePlot.R")

images=read.csv("../../data/valid.csv",nrows=10000) 

images_for_special_label=colMeans(subset(images,label==4))   # [1:785]

average_matrix=matrix(ceiling(images_for_special_label[2:785]),c(28,28),byrow=TRUE) 

#display the image 
myImagePlot(average_matrix) 
