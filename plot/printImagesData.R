source("./myImagePlot.R")
images=read.csv("../data/valid.csv",nrows=10) 

#get the 23th and cast to matrix
b=as.matrix(images[6,2:785],rownames.force=NA) 

b=matrix(b,28,28,byrow=TRUE) 

#display the image 
myImagePlot(b) 
