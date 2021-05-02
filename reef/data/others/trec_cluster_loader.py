#!/usr/bin/env python
import numpy as np
import scipy
import json
from sklearn import model_selection as cross_validation
import re
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans

def parse_file(filename):

    def parse(filename):
        tweet = []
        print(filename)
        with open(filename) as f:
            for line in f:
#                 print(line)
                tweet.append(line)
        return tweet

    tweets = parse(filename)
    gt = []
    plots = []
    idx = []
    for i,twt in enumerate(tweets):
        tweet = twt.split(':')
#         print(tweet)
        genre = tweet[0]
        tweet_txt = tweet[1]
#         tweet_txt = re.sub(r"@\w+","", tweet[1])
#         tweet_txt = ' '.join(tweet_txt.split(' ')[3:])
        
        if 'NUM' in genre:
            plots.append(tweet_txt)
            gt.append(0)
            idx.append(i)
        elif 'LOC' in genre:
            plots.append(tweet_txt)
            gt.append(1)
            idx.append(i)
        elif 'HUM' in genre:
            plots.append(tweet_txt)
            gt.append(2)
            idx.append(i)
        elif 'DESC' in genre:
            plots.append(tweet_txt)
            gt.append(3)
            idx.append(i)
        elif 'ENTY' in genre:
            plots.append(tweet_txt)
            gt.append(4)
            idx.append(i)
        elif 'ABBR' in genre:
            plots.append(tweet_txt)
            gt.append(5)
            idx.append(i)
        else:
            continue  

    print('len of data',len(plots))
    return np.array(plots), np.array(gt)

def split_data(X, plots, y):
    np.random.seed(1234)
    num_sample = np.shape(X)[0]
    num_test = 500

    X_test = X[0:num_test,:]
    X_train = X[num_test:, :]
    plots_train = plots[num_test:]
    plots_test = plots[0:num_test]

    y_test = y[0:num_test]
    y_train = y[num_test:]

    # split dev/test
    test_ratio = 0.2
    X_tr, X_te, y_tr, y_te, plots_tr, plots_te =cross_validation.train_test_split(X_train, y_train, plots_train, test_size = test_ratio, random_state=25)

    return np.array(X_tr.todense()), np.array(X_te.todense()), np.array(X_test.todense()),\
     np.array(y_tr), np.array(y_te), np.array(y_test), plots_tr, plots_te, plots_test


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

    def load_data(self, dataset, data_path='/home/ayusham/Semi_Supervised_LFs/Data/TREC/', cls =0):
     
        plots, labels = parse_file(data_path+'all.txt')
        
        #Featurize Plots  
        vectorizer = CountVectorizer(min_df=1, binary=True, \
            decode_error='ignore', strip_accents='ascii', ngram_range=(1,2))
        X = vectorizer.fit_transform(plots)
        valid_feats = np.where(np.sum(X,0)> 2)[1]
        X = X[:,valid_feats]
        
        kmeans = KMeans(n_clusters=6, random_state=25).fit(X)
        X = X[np.where(kmeans.labels_ == cls)]
        plots = plots[np.where(kmeans.labels_ == cls)]
        labels = labels[np.where(kmeans.labels_ == cls)]
        print('len of labels', len(labels))
#         Split Dataset into Train, Val, Test
        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
        train_ground, val_ground, test_ground, train_plots , val_plots, test_plots = split_data(X, plots, labels)

         
        #Prune Feature Space
        common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix)
#         print('common_idx',len(common_idx))
        return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix[:,common_idx],             np.array(train_ground), np.array(val_ground), np.array(test_ground),train_plots, val_plots, test_plots

