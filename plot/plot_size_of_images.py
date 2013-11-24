import cv2
import os
import sys
import time
import random
import histogram2d

def simple_browse(dir_path, limit=0):
    num=0
    h_stat=[]
    w_stat=[]
    dir_list = os.walk(dir_path)
    for root, dirs, files in dir_list:
        for f in files:
            filename=os.path.join(root, f)
            num+=1
            print num
            img=cv2.imread(filename)
            h, w, layers =  img.shape
            h_stat.append(h)
            w_stat.append(w)
            print h,w,layers
            if limit==num:
                break
    print "total:",len(h_stat)
    return (h_stat,w_stat)

if __name__=="__main__":
    if len(sys.argv)!=2:
        print  ''' USAGE: %s <images_path> ''' % sys.argv[0]
        sys.exit(0)
    dir_path=sys.argv[1]
    h_list,w_list=simple_browse(dir_path,0)
    histogram2d.scatter_hist(h_list,w_list)

    #points, sub = histogram2d.hist2d_bubble(w_list, h_list, bins=10)
    #sub.axes.set_xlabel('image_width')
    #sub.axes.set_ylabel('image_height')
