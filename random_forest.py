import fasttext
import numpy as np
import helpers
import nltk
import sklearn
from sklearn.ensemble import RandomForestClassifier

PATH_POS = 'twitter-datasets/train_pos.txt'
PATH_NEG = 'twitter-datasets/train_neg.txt'
PATH_OUT = 'twitter-datasets/train_combine.txt'

data_pos = ""
data_neg = ""

with open(PATH_POS) as fp: 
    data_pos = fp.read() 
    
with open(PATH_NEG) as fp: 
    data_neg = fp.read() 
    
data = data_pos + data_neg 
  
with open (PATH_OUT, 'w') as fp: 
    fp.write(data) 

PATH_TRAIN_DATA = 'twitter-datasets/train_combine.txt'
dimension = 100

PATH_TEST_DATA = 'twitter-datasets/test_data.txt'

ids, querys = helpers.read_test(PATH_TEST_DATA)
train_data = helpers.read_train(PATH_TRAIN_DATA)

model = fasttext.train_unsupervised(PATH_TRAIN_DATA, model = 'cbow', dim=dimension)
vocabulary = model.words
word_embeddings = np.array([model[word] for word in vocabulary])

def compute_word_embedding(model, data, dimension):
    ret_word_embedding = np.zeros((len(data), dimension))

    for i, sentence in enumerate(data):
        words = sentence.split(sep = ' ')
        count = 0
        avg_word_vector = np.zeros(dimension)
        for word in words:
            if word in vocabulary:
                avg_word_vector = np.add(avg_word_vector,model[word])
                count += 1
        if count != 0:
            avg_word_vector = avg_word_vector / count
        ret_word_embedding[i] = avg_word_vector
    
    return ret_word_embedding

x_test = compute_word_embedding(model, querys, dimension)
x_train = compute_word_embedding(model, train_data, dimension)
y_train = [1] * 100000 + [0] * 100000 # change the number to 1250000 if you want to use full dataset

# train on random forest
print('start training RF')
classifier = RandomForestClassifier(n_estimators=300, random_state=0)
classifier.fit(x_train, y_train)
print('train finish')
y_pred = classifier.predict(x_test)

y_pred = np.where(y_pred==0, -1, y_pred)
OUTPUT_PATH = 'twitter-datasets/submission.csv'

helpers.create_csv_submission(ids, y_pred, OUTPUT_PATH)
print('Done!')