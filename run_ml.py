import numpy as np
from helpers import *
from data_preprocess import *
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

# get preprocessed data
ids, querys = read_test(PATH_TEST_DATA)
train_data = read_train(PATH_TRAIN_DATA)

print("creating word embedding...")
# we use continuous bag of words for word embedding
dimension = 100
model = fasttext.train_unsupervised(PATH_TRAIN_DATA, model = 'cbow', dim=dimension)
vocabulary = model.words
word_embeddings = np.array([model[word] for word in vocabulary])

# create our final training and testing data
x_test = compute_word_embedding(model, querys, dimension, vocabulary)
x_train = compute_word_embedding(model, train_data, dimension, vocabulary)
y_train = [1] * 1250000 + [0] * 1250000 # change the number to 1250000 if you want to use full dataset

# choose one simple ml classifier and train model
# gaussian naive bayes
classifier = GaussianNB()
# random forest
#classifier = RandomForestClassifier(n_estimators=300)
# SVM
#classifier = make_pipeline(StandardScaler(), SVC(gamma='auto'))

print('Start training...')
classifier.fit(x_train, y_train)
print('train finish')

# create prediction
y_pred = classifier.predict(x_test)
y_pred = np.where(y_pred==0, -1, y_pred)
OUTPUT_PATH = 'twitter-datasets/submission.csv'
create_csv_submission(ids, y_pred, OUTPUT_PATH)
print('Done!')
