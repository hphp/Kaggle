#!/usr/bin/python
import cv2
import sys
import logging
import time
from numpy import arange
from data_browsers import browse_images


verbosity = logging.INFO
logging.basicConfig(filename=None,level=verbosity,)

from DogVsCat_lenet_for_detect import image_recognition

wind_name="detecting"
cv2.namedWindow(wind_name)
cv2.moveWindow(wind_name, 800,0)    

def detectByMuitScaleSlideWindows(img,windowSize=(15,15),wStep=5,hStep=5,classifier=None):
    dog_rects=[]
    cat_rects=[]
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
                rect = (x,y,patchWidth,patchHeight)
                show_rectangle(img,rect)
                subImg=img[y:y+patchHeight,x:x+patchWidth]
                #if classifier.isDog(subImg):
                the_label=classifier.Recognize(subImg)
                if the_label==DogClassifier.Dog:
                    print "found one dog:",rect
                    dog_rects.append(rect)
                elif the_label==DogClassifier.Cat:
                    print "found one cat:",rect
                    cat_rects.append(rect)
                    
    logging.info("total checked:%s",cnt)
    return (dog_rects,cat_rects)

def detectObject(img):
    """
    This should be pure opencv and reasonably quick.
    It carrys out the actual detection, and returns a list of objects found
    """
    objects=[(20,20,50,10),(30,50,20,40)]
    return objects

def show_rectangle(img, rect,color=(255,0,255)):
    #time.sleep(2)
    img_for_draw= img.copy()
    x,y,w,h=rect
    cv2.rectangle(img_for_draw, (x,y), (x+w,y+h), color, 1)
    #cv2.rectangle(img_for_draw, (x,y), (x+w,y+h), (randint(0,255),0,randint(0,255)), 1)
    cv2.imshow( "detecting", img_for_draw)
    #time.sleep(0.1)
    if 27==cv2.waitKey(1):
        sys.exit(0)    



def detect_and_draw( img):
    """
    draw a box with opencv on the image around the detected faces and display the output
    """
    from random import randint 
    start_time = cv2.getTickCount()
    dog_classifier=DogClassifier()
    print dog_classifier
    img=cv2.resize(img,(200,200))
    dogs,cats = detectByMuitScaleSlideWindows(img,windowSize=(50,50),classifier=dog_classifier)
    end_time = cv2.getTickCount() 
    logging.info("time cost:%gms",(end_time-start_time)/cv2.getTickFrequency()*1000.)
    #dogs = detectObject(img)
    print "found dogs:",len(dogs)
    max_rect=(0,0,0,0)
    for rect in dogs:
        x,y,w,h=rect
        if w*h>max_rect[2]*max_rect[3]:
            max_rect=rect
    x,y,w,h=max_rect
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 1)

    print "found cats:",len(cats)
    max_rect=(0,0,0,0)
    for rect in cats:
        x,y,w,h=rect
        if w*h>max_rect[2]*max_rect[3]:
            max_rect=rect
    x,y,w,h=max_rect
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 1)
    return img
            

def printDetail(img):
    print dir(img)
    print img

def main():
    """Run the default object detector"""
    #printDetail(img)


class DogClassifier():
    Dog=1
    Cat=0
    Other=2 
    def __init__(self):
        self.label_type = [0, 0, 0]
        pass

    def isDog(self,img):
        label=image_recognition(img)
        self.label_type[label] += 1
        print self.label_type
        print label
        return (1==label)
    
    def Recognize(self,img):
        label=image_recognition(img)
        return label
        
        

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print  ''' USAGE: %s <images_path> [simple|random]''' % sys.argv[0]
        sys.exit(0)
    dir_path=sys.argv[1]
    #dir_path='/Users/xcbfreedom/projects/data/dogs_vs_cats'
    if sys.argv[2]=="simple":
        browse_images.simple_browse(dir_path,img_handle=detect_and_draw)
    elif sys.argv[2]=="random":
        browse_images.random_browse(dir_path)
    #(img)
