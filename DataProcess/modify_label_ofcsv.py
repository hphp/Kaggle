'''
2013-12-02
modify labels in existing feature.csv s
'''

import transform_data_to_format as tdtf

DataHome = "../../data/Kaggle/DogVsCatData/"
from_feature_filname = ""
to_feature_filename = DataHome + "DogVsCat_bghead_valid_feature_2c_2500.csv"
from_feature_filename = DataHome + "DogVsCat_head_valid_feature_3c_2500.csv"

mod_labels = [[0,0],[1,0],[2,1]]

f_feature_list = tdtf.read_feature_from_csv(filname=from_feature_filename, limit=None, header_n=0)
t_feature_list = []
for feature in f_feature_list:
    for pair in mod_labels:
        if feature[0] == pair[0]:
            feature[0] = pair[1]
            break
    t_feature_list.append(feature)
tdtf.wr_content_to_csv(t_feature_list, to_feature_filename)
