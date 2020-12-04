# -*- coding: utf-8 -*-
import csv
import numpy as np
import datetime
import fasttext

def read_test(path):
    """read test data"""
    with open(path) as fp:
        data_test = fp.readlines()

    ids = [0] * len(data_test)
    for i in range(len(data_test)):
        ids[i] = data_test[i].split(',', 1)[0]
        data_test[i] = data_test[i].split(',', 1)[1]
    
    return ids, data_test

def read_train(path):
    """read train data"""
    with open(path) as fp:
        data_train = fp.readlines()
    return data_train

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def calculate_time(elapsed):
    """
    Takes a time in seconds and returns format as hh:mm:ss
    """

    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def batch_accuracy(preds, labels):
    """Function to calculate the accuracy of predictions vs labels"""

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def compute_word_embedding(model, data, dimension, vocabulary):
    """Function to compute word embedding(bag of words)"""
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
