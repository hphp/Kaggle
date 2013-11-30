import cv2
import os
import sys
import re
import time
import random

#height , width , layers =  img1.shape

def resize(img):
    height , width =(200,200)
    img=cv2.resize(img, (width,height))
    return img

def isOK(filename):
    return re.match(r'^.*\.jpg$',filename)

def simple_browse(dir_path, img_handle=None):
    cv2.namedWindow("Example1")
    num=0
    dir_list = os.walk(dir_path)
    for root, dirs, files in dir_list:
        for f in files:
            filename=os.path.join(root, f)
            if not isOK(filename):
                print 'skip illegal file:',filename
                continue
            num+=1
            print num 
            img=cv2.imread(filename)
            if img_handle:
                img=img_handle(img)
            cv2.imshow( "Example1", img)
            if 27==cv2.waitKey(500):
                break
    cv2.destroyAllWindows()

def random_browse(images_dir, img_handle=None):
    cv2.namedWindow("Example1")
    dir_list = os.walk(images_dir)
    num=0
    for root, dirs, files in dir_list:
        while True:
            f=files[random.randint(0,len(files)-1)]
            filename=os.path.join(root, f)
            if not isOK(filename):
                print 'skip illegal file:',filename
                continue
            img=cv2.imread(filename)
            if img_handle:
                img=img_handle(img)
            num+=1
            print num
            cv2.imshow( "Example1", img)
            if 27==cv2.waitKey(500):
                break
        break  # just handle the files in top dir.
    cv2.destroyAllWindows()
if __name__=="__main__":
    if len(sys.argv)!=3:
        print  ''' USAGE: %s <images_path> [simple|random]''' % sys.argv[0]
        sys.exit(0)
    dir_path=sys.argv[1]
    #dir_path='/Users/xcbfreedom/projects/data/dogs_vs_cats'
    if sys.argv[2]=="simple":
        simple_browse(dir_path)
    elif sys.argv[2]=="random":
        random_browse(dir_path)
        #random_browse(dir_path,img_handle=resize)
