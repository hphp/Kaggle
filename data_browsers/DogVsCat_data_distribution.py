
import os
DataHome = "../../data/Kaggle/DogVsCatData/"

img_name_list = os.listdir(DataHome + "train/")
dog_count = cat_count = 0
other_count = 0
for img_name in img_name_list:
    name_part_list = img_name.split(".")
    if name_part_list[0] == "dog":
        dog_count += 1
    elif name_part_list[0] == "cat":
        cat_count += 1
    else:
        other_count += 1
print "dog image number : ", dog_count
print "cat image number : ", cat_count
print "other image number : ", other_count
