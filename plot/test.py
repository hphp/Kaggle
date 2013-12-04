import math, datetime
import time
import rpy2.robjects.lib.ggplot2 as ggplot2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.interactive import process_revents
grdevices = importr('grDevices')
process_revents.start()

base = importr('base')
datasets= importr('datasets')

mtcars = datasets.__rdata__.fetch('mtcars')['mtcars']
pp = ggplot2.ggplot(mtcars) +  ggplot2.aes_string(x='wt', y='mpg', col='factor(cyl)') +  ggplot2.geom_point() +  ggplot2.geom_smooth(ggplot2.aes_string(group = 'cyl'), method = 'lm') 
#pp.plot()
#process_revents.start()
print(pp)
process_revents.process_revents()

while True:
    time.sleep(1)
process_revents.stop()
