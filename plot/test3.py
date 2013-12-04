import time
#import cv2
from ggplot import *
pp=ggplot(mtcars, aes('mpg', 'qsec')) +  \
            geom_point(colour='steelblue') + \
            scale_x_continuous(breaks=[10,20,30],  \
            labels=["horrible", "ok", "awesome"])
#print dir(pp)
#pp.draw()
print pp
plt.show(1)
#cv2.waitKey(1)
#while 1:
#    time.sleep(1)
