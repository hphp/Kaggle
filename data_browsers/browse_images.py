import cv2
import os
import sys
import time
import random

#height , width , layers =  img1.shape

def simple_browse(dir_path):
    cv2.namedWindow("Example1")
    num=0
    height , width =(200,200)
    dir_list = os.walk(dir_path)
    for root, dirs, files in dir_list:
        for f in files:
            filename=os.path.join(root, f)
            num+=1
            print num
            img=cv2.imread(filename)
            #img=cv2.resize(img, (width,height))
            cv2.imshow( "Example1", img)
            if 27==cv2.waitKey(500):
                break


def random_browse(images_dir):
    cv2.namedWindow("Example1")
    height , width =(200,200)
    dir_list = os.walk(images_dir)
    num=0
    for root, dirs, files in dir_list:
        while True:
            f=files[random.randint(0,len(files)-1)]
            filename=os.path.join(root, f)
            img=cv2.imread(filename)
            #img=cv2.resize(img, (width,height))
            num+=1
            print num
            cv2.imshow( "Example1", img)
            if 27==cv2.waitKey(500):
                break
        break  # just handle the files in top dir.

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
