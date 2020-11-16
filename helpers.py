# -*- coding: utf-8 -*-
import csv
import numpy as np

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
