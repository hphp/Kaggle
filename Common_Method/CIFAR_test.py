
import sys
import cv2
import numpy
if not "../Common_Method/" in sys.path:
    sys.path.append("../Common_Method/")
if not "../DataProcess/" in sys.path:
    sys.path.append("../DataProcess/")
import transform_data_to_format as tdtf

import lenet_test
import os

DataHome = "../../data/Kaggle/CIFAR-10/"
ModelHome = DataHome #"../trained_model/"
train_model_route = ModelHome + "CIFAR_lenet_0.15_w41_ep100.np.pkl"

lenet_test.DataHome = DataHome
lenet_test.ModelHome = ModelHome
lenet_test.train_model_route = train_model_route

def raw_img_recognition(img_route):
    img = cv2.imread(img_route)
    #print "orignal img: ",type(img),img.shape,img.dtype
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print img.size

    # show
    #cv2.imshow("w1",img)
    #cv2.waitKey(0)

    # do recognition
    img = numpy.asarray(img,dtype='float32') / 256.
    img_label = lenet_test.raw_img_recognition(img)
    #img_label = 0
    return img_label

def get_img_id(filename):
    file_part = filename.split(".")
    return file_part[0]

if __name__ == '__main__':
    test_DataHome = DataHome + "test/"
    test_filename_list = os.listdir(test_DataHome)
    answer_dict = {}
    cnt = 0
    for test_filename in test_filename_list:
        img_label = raw_img_recognition(test_DataHome + test_filename)
        img_id = get_img_id(test_filename)
        answer_dict[int(img_id)] = img_label
        cnt += 1
        if (cnt % 1000) == 0:
            print "current ", cnt

    id_list = []
    pred_list = []
    for i in range(1,300001):
        if i not in answer_dict:
            continue
        id_list.append(i)
        pred_list.append(answer_dict[i])

    print pred_list 
    tdtf.wr_to_csv(['id','label'], id_list, pred_list, DataHome + "CIFAR_lenet_0.15_w41_ep100.csv")
