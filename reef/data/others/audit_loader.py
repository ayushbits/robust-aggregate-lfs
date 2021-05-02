#!/usr/bin/env python


import numpy as np
import scipy
import json
from sklearn import model_selection as cross_validation
import re
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle

def parse_file(filename):

    def parse(filename):
        tweet = []
        with open(filename) as f:
            for line in f:
                tweet.append(line)
        return tweet

    tweets = parse(filename)
    gt = []
    plots = []
    idx = []
    for i,twt in enumerate(tweets):
        tweet = twt.split(',')
        # print(tweet)
        genre = tweet[1]
        tweet_txt = re.sub(r"@\w+","", tweet[10])
        
        if 'neutral' in genre:
            continue
        elif 'positive' in genre:
            plots.append(tweet_txt)
            gt.append(1)
            idx.append(i)
        elif 'negative' in genre:
            plots.append(tweet_txt)
            gt.append(-1)
            idx.append(i)
        else:
            continue  

    print(len(plots))
    return np.array(plots), np.array(gt)

def split_data(X, y):
    np.random.seed(1234)
    num_sample = np.shape(X)[0]
    num_test = 500
    X, y  = shuffle(X, y, random_state = 25)

    X_test = X[0:num_test,:]
    X_train = X[num_test:, :]
#     plots_train = plots[num_test:]
#     plots_test = plots[0:num_test]

    y_test = y[0:num_test]
    y_train = y[num_test:]

    # split dev/test
    test_ratio = 0.2
    X_tr, X_te, y_tr, y_te=  cross_validation.train_test_split(X_train, y_train, test_size = test_ratio, random_state=25)

    return np.array(X_tr), np.array(X_te), np.array(X_test),  np.array(y_tr), np.array(y_te), np.array(y_test)


class DataLoader(object):
    """ A class to load in appropriate numpy arrays
    """

    def prune_features(self, val_primitive_matrix, train_primitive_matrix, thresh=0.01):
        val_sum = np.sum(np.abs(val_primitive_matrix),axis=0)
        train_sum = np.sum(np.abs(train_primitive_matrix),axis=0)

        #Only select the indices that fire more than 1% for both datasets
        train_idx = np.where((train_sum >= thresh*np.shape(train_primitive_matrix)[0]))[0]
        val_idx = np.where((val_sum >= thresh*np.shape(val_primitive_matrix)[0]))[0]
        common_idx = list(set(train_idx) & set(val_idx))

        return common_idx

    def load_data(self, dataset, data_path='/home/ayusham/Semi_Supervised_LFs/dsets/audit/'):
        #Parse Files
#         plots, labels = parse_file(data_path+'risk.csv')
        #read_plots('imdb_plots.tsv')

        #Featurize Plots  
#         vectorizer = CountVectorizer(min_df=1, binary=True,   decode_error='ignore', strip_accents='ascii', ngram_range=(1,2))
        # vectorizer = TfidfVectorizer(min_df=1, binary=True, \
        #     decode_error='ignore', strip_accents='ascii', ngram_range=(1,2))
        
#         X = vectorizer.fit_transform(plots)
        X1 = np.load(data_path+'train_features.npy')
        X2 = np.load(data_path+'test_features.npy')
        labels1 = np.load(data_path+'train_y.npy')
        labels2 = np.load(data_path+'test_y.npy')
        X = np.concatenate((X1,X2))
        labels = np.concatenate((labels1, labels2))

#        valid_feats = np.where(np.sum(X,0)> 2)[1]
#        X = X[:,valid_feats]
        labels = np.asarray(labels)
        labels[labels==0] = -1

        #Split Dataset into Train, Val, Test

        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            train_ground, val_ground, test_ground  = split_data(X, labels)
        train_plots = val_plots = test_plots = []
        #Prune Feature Space
        common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix)
        return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix[:,common_idx],             np.array(train_ground), np.array(val_ground), np.array(test_ground), train_plots, val_plots, test_plots

