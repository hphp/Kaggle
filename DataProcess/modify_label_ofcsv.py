'''
2013-12-02
modify labels in existing feature.csv s
'''

import transform_data_to_format as tdtf

DataHome = "../../data/Kaggle/DogVsCatData/"
to_feature_filename = DataHome + "DogVsCat_bghead_valid_feature_2c_2500.csv"
from_feature_filename = DataHome + "DogVsCat_head_valid_feature_3c_2500.csv"
DataHome = "../../data/Kaggle/CIFAR-10/"
#to_feature_filename = DataHome + "trainIdCls.csv"
#from_feature_filename = DataHome + "trainLabels.csv"
mod_label_filename = DataHome + "trainIdCls.csv"
to_feature_filename = DataHome + "train_feature_Cls_pixelv.csv"
from_feature_filename = DataHome + "train_feature_pixel_v.csv"

mod_labels = [[0,0],[1,0],[2,1]]
#awk -F ',' '{a[$2]++}END{for(i in a)if(i!="label")print ",[\""i"\",\""b++"\"]"}' ~/Documents/data/Kaggle/CIFAR-10/trainLabels.csv
mod_labels = [
["automobile","0"]
,["frog","1"]
,["bird","2"]
,["dog","3"]
,["horse","4"]
,["airplane","5"]
,["cat","6"]
,["truck","7"]
,["deer","8"]
,["ship","9"]
]
mod_labels = tdtf.read_s_feature_from_csv(filname=mod_label_filename, limit=None, header_n=1)

f_feature_list = tdtf.read_s_feature_from_csv(filname=from_feature_filename, limit=10, header_n=0)
t_feature_list = []
print len(f_feature_list)
for feature in f_feature_list:
    for pair in mod_labels:
        if feature[0] == str(pair[0]):
            feature[0] = pair[1]
            break
    t_feature_list.append(feature)

index = 0
for t_feature in t_feature_list:
    print t_feature[0:2]
    index += 1
    if index > 10:
        break
#tdtf.wr_content_to_csv(t_feature_list, to_feature_filename)
