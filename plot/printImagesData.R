source("./myImagePlot.R")

images=read.csv("../../data/valid.csv",nrows=10) 

#get the 23th and cast to matrix
b=as.matrix(images[6,2:785],rownames.force=NA) 

b=matrix(b,28,28,byrow=TRUE) 

#display the image 
myImagePlot(b) 



den=density(images$pixel622)
hist(images$pixel622,xlim=range(c(den$x,0,256)),ylim=range(c(den$y,20)))
#hist(images$pixel622)
par(new=T)   # not to clear the plane
plot(den,main="",xlab="",ylab="",col="red")