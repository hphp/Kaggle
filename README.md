
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

./plot  ---  some tools for data plot

./reports  -- some test report or plot results 
==========================================
