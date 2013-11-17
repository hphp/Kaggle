#!/usr/bin/python
import cv2
import sys
import logging
import time
from numpy import arange

verbosity = logging.INFO
logging.basicConfig(filename=None,level=verbosity,)


def detectByMuitScaleSlideWindows(img,windowSize=(15,15),wStep=5,hStep=5,classifier=None):
    positive_rects=[]
    imgHeight=img.shape[0]
    imgWidth=img.shape[1]
    logging.info("img size. width:%s,height:%s",imgWidth,imgHeight)
    max_scale=int(min(imgWidth*1.0/windowSize[0],imgHeight*1.0/windowSize[1]))
    logging.info("max_scale:%s",max_scale)
    scale_step=0.5
    cnt=0
    for scale in arange(1,max_scale+scale_step,scale_step):
        patchWidth=int(windowSize[0]*scale)
        patchHeight=int(windowSize[1]*scale)
        #the check points
        xSpot=(imgWidth-patchWidth)/wStep+1
        ySpot=(imgHeight-patchHeight)/wStep+1

        #----
        xRange=range(0,imgWidth-patchWidth+1,wStep) 
        yRange=range(0,imgHeight-patchHeight+1,hStep)
        #--also can use
        #xRange=range(0, xSpot*wStep, wStep)
        #yRange=range(0, ySpot*hStep, hStep)
        #------

        logging.info("scale:%s, patchWidth:%s, patchHeight:%s, xRange [%s,%s], yRange [%s,%s], xSpot:%s, ySpot:%s, count:%s", scale, patchWidth, patchHeight,xRange[0],xRange[-1],yRange[0],yRange[-1],xSpot,ySpot,xSpot*ySpot)
        for x in xRange:
            for y in yRange:
                cnt+=1
                #subImg=img[y:y+patchHeight,x:x+patchWidth]
                #if classifier.isDog(subImg):
                #print (x,y,patchWidth,patchHeight)
                positive_rects.append((x,y,patchWidth,patchHeight))
    logging.info("total:%s",cnt)
    return positive_rects

def detectObject(img):
    """
    This should be pure opencv and reasonably quick.
    It carrys out the actual detection, and returns a list of objects found
    """
    objects=[(20,20,50,10),(30,50,20,40)]
    return objects

def detect_and_draw( img):
    """
    draw a box with opencv on the image around the detected faces and display the output
    """
    from random import randint 
    start_time = cv2.getTickCount()
    objects = detectByMuitScaleSlideWindows(img)
    end_time = cv2.getTickCount() 
    logging.info("time cost:%gms",(end_time-start_time)/cv2.getTickFrequency()*1000.)
    #objects = detectObject(img)
    if objects:
        for x,y,w,h in objects:
            #time.sleep(2)
            img_for_draw= img.copy()
            cv2.rectangle(img_for_draw, (x,y), (x+w,y+h), (255,0,255), 1)
            #cv2.rectangle(img_for_draw, (x,y), (x+w,y+h), (randint(0,255),0,randint(0,255)), 1)
            cv2.imshow( "result", img_for_draw)
            #time.sleep(0.1)
            cv2.waitKey(1)


def printDetail(img):
    print dir(img)
    print img

def main():
    """Run the default object detector"""
    import sys
    from glob import glob
    import itertools as it
    if len(sys.argv)!=2:
        print  ''' USAGE: xx.py <image_file_name> '''
        sys.exit(0)
    img = cv2.imread(sys.argv[1])
    #printDetail(img)
    detect_and_draw(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
