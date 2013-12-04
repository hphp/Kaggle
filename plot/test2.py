from rpy2 import robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.lib.ggplot2 as ggplot2
import time
r = robjects.r
grdevices = importr('grDevices')
p = r('''
  library(ggplot2)
    p <- ggplot(diamonds, aes(clarity, fill=cut)) + geom_bar()
      p <- p + opts(title = "{0}")
      # add more R code if necessary e.g. p <- p + layer(..)
        p'''.format("stackbar")) 
      # you can use format to transfer variables into R
        # use var.r_repr() in case it involves a robject like a vector or data.frame
#print p
p.plot()
#r'dev.off')
r.dev_off()
while 1:
    time.sleep(1)
#p.plot()
#ggplot2.show(1)
