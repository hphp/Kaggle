
====setup===
1. add Kaggle paht to PYTHONPATH


============= directory tree ======
../data 
    Kaggle
        DogVsCatData
            head_images/ -- head jpg files
            bg_images/ -- background jpg files
            train/ -- jpg files
            test1/ -- jpg files

            *_feature_*.csv -- feature csv file
            *_trained_model_*.pkl -- trained model

        dog == 1
        cat == 0
            
        MNISTData
            train.csv -- train data from Kaggle
            test.csv -- test data from Kaggle
            valid.csv -- valid data from Kaggle

            METHOD*_trained_model.pkl -- trained model using METHOD*
            pringle_METHOD*.csv     -- Kaggle submission data using METHOD*

        CIFAR-10
            ##### original image ######
            train/ -- 50000 train pictures of size 32*32 (color) , named:Id.jpg
            test/ -- 100000 test pictures of size 32*32

            ##### label-class transform ######
            trainLabels.csv -- Id,Label all 50000 train Ids.
            trainIdCls.csv -- Id,Cls all 50000 train Ids

            ##### raw pixel-(Id/Class) info ######
            train_feature_pixel_v.csv -- Id,pixel[0][0],pixel[0][1],....,50000
            train_feature_Cls_pixelv.csv -- Cls,pixel[0][0],...., 50000

            ##### features that are used to train/test ######
            CIFAR_train_feature.csv -- Cls,pixel[0][0] ,.. , selected randomly from train_feature_Cls_pixelv.csv , 25000
            CIFAR_valid_feature.csv -- Cls,pixel[0][0] ,.. , selected randomly from train_feature_Cls_pixelv.csv , 25000

            CIFAR_train4_feature.csv -- Cls,pixel[0][0] ,.. , selected randomly from train_feature_Cls_pixelv.csv , 40000
            CIFAR_valid1_feature.csv -- Cls,pixel[0][0] ,.. , selected randomly from train_feature_Cls_pixelv.csv , 10000

            ##### trained model ######
            ##### Project_Method_LearningRate_FeatureProperty_EpoNumber ######
            CIFAR_lenet_0.05_w41_ep5000.np.pkl -- w41: whole dataset and train : valid = 4:1
            CIFAR_lenet_0.13_w41_ep5000.np.pkl -- w41: whole dataset and train : valid = 4:1

            ##### test_output ######
            ##### Id,ClassId ######
            CIFAR_lenet_0.13_w41_ep5000.csv

            ##### Kaggle final upload file ######
            CIFAR_upload_lenet_0.13_w41_ep5000.csv

./plot  ---  some tools for data plot

./reports  -- some test report or plot results 
==========================================
