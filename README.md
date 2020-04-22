''' @author : Durgesh Chalvadi '''

# Generalized_ML_Framework

  export TRAINING_DATA=input/train_folds.csv
  export TEST_DATA=input/test_cat.csv
  export MODEL=$1

# Training model using stratifiedkfold

  FOLD=0 python -m src.train
  FOLD=1 python -m src.train
  FOLD=2 python -m src.train
  FOLD=3 python -m src.train
  FOLD=4 python -m src.train

# Get Predictions
  python -m src.predict  


# Steps to run 
  sh run.sh randomforest  # randomforest is an cmd line argument specifying which classifier to be used.refer dispatcher.py

# Running  Logistic Regression for categorical data
 python categorical.py

