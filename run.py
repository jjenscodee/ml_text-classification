import numpy as np
from xlnet import *
from helpers import *
from data_preprocess import *
from numpy import asarray
from bert import *

print("Loading data...")
# load data and combine train_pos + train_neg
PATH_POS = 'twitter-datasets/train_pos_full.txt'
PATH_NEG = 'twitter-datasets/train_neg_full.txt'
PATH_TEST = 'twitter-datasets/test_data.txt'
PATH_COMBINE = 'twitter-datasets/train_combine.txt'

data_pos = ""
data_neg = ""

with open(PATH_POS) as fp: 
    data_pos = fp.read() 
    
with open(PATH_NEG) as fp: 
    data_neg = fp.read() 
  
data = data_pos + data_neg 
  
with open (PATH_COMBINE, 'w') as fp: 
    fp.write(data)

print("preprocessing data...")
# data preprocess
PATH_TRAIN_DATA = "twitter-datasets/train_preprocess"
PATH_TEST_DATA = "twitter-datasets/test_preprocess"
data_process(PATH_COMBINE, PATH_TRAIN_DATA, PATH_TEST, PATH_TEST_DATA)

# get our training & testing data
train_data = read_train(PATH_TRAIN_DATA)
ids, test_data = read_test(PATH_TEST_DATA)

x_train = np.array(train_data)
x_test = np.array(test_data)

# change the number to 1250000 if you want to use full dataset
y_train = [1] * 1250000 + [0] * 1250000 

print("Start training...")
# train XLnet and get the results
#flat_predictions = train_xlnet(x_train=x_train, y_train=y_train, batch_size=32, lr=2e-5, epochs=3, ids=ids, x_test=x_test)

# train Bert and get the results
flat_predictions = train_bert(x_train=x_train, y_train=y_train, batch_size=32, lr=2e-5, epochs=2, ids=ids, x_test=x_test)


#change 0 to -1
predictions = np.where(flat_predictions==0, -1, flat_predictions)

# create submission
OUTPUT_PATH = 'twitter-datasets/submission.csv'
create_csv_submission(ids, predictions, OUTPUT_PATH)
