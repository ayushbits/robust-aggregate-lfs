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
    X, plots, y  = shuffle(X, plots, y, random_state = 25)
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

    def prune_features(self, val_primitive_matrix, train_primitive_matrix, thresh=0.001):
        val_sum = np.sum(np.abs(val_primitive_matrix),axis=0)
        train_sum = np.sum(np.abs(train_primitive_matrix),axis=0)

        #Only select the indices that fire more than 1% for both datasets
        train_idx = np.where((train_sum >= thresh*np.shape(train_primitive_matrix)[0]))[0]
        val_idx = np.where((val_sum >= thresh*np.shape(val_primitive_matrix)[0]))[0]
        common_idx = list(set(train_idx) & set(val_idx))

        return common_idx

    def load_data(self, dataset, data_path='/home/ayusham/Semi_Supervised_LFs/Data/TREC/'):
     
        plots, labels = parse_file(data_path+'all.txt')
        def mytokenizer(text):
            return text.split()
        
        #Featurize Plots  
    # niche ki  line original code ka bhag hai
    
#         vectorizer = CountVectorizer(min_df=1, binary=True, stop_words='english', \
#             decode_error='ignore', strip_accents='ascii', ngram_range=(1,2))

        vocab = {'abbreviation':0,'actor':1,'actress':2,'address':3,'age':4,'alias':5,'amount':6,'are':7,'around':8,'at':9,'ate':10,'book':11,'build':12,'built':13,'by':14,'called':15,'can':16,'capital':17,'captain':18,'citizen':19,'close':20,'company':21,'composed':22,'could':23,'country':24,'date':25,'day':26,'demands':27,'describe':28,'did':29,'discovered':30,'division':31,'do':32,'doctor':33,'does':34,'does ':35,'doesn':36,'engineer':37,'enumerate':38,'explain':39,'far':40,'fastener':41,'fastener ':42,'fear':43,'for':44,'found':45,'from':46,'game':47,'gamer':48,'governs':49,'group':50,'groups':51,'guarded':52,'hero':53,'hours':54,'how':55,'human':56,'hypertension':57,'in':58,'instance':59,'invented':60,'is':61,'is ':62,'island':63,'kind':64,'king':65,'latitude':66,'latitude ':67,'lawyer':68,'leader':69,'leads':70,'list':71,'lived':72,'lives':73,'located':74,'long':75,'longitude':76,'made':77,'man':78,'many':79,'mean':80,'meant':81,'minute':82,'model':83,'month':84,'movie':85,'much':86,'name':87,'name ':88,'nationalist':89,'near':90,'nicknamed':91,'novel':92,'number':93,'object':94,'of':95,'old':96,'organization':97,'origin':98,'out':99,'owner':100,'owns':101,'part':102,'patent':103,'pays':104,'percentage':105,'person':106,'play':107,'played':108,'player':109,'poet':110,'population':111,'portrayed':112,'president':113,'queen':114,'ratio':115,'run':116,'seconds':117,'served':118,'shall':119,'share':120,'short':121,'should':122,'should ':123,'situated':124,'slept':125,'small':126,'speed':127,'stand':128,'star':129,'studied':130,'study ':131,'surname':132,'surrounds':133,'take':134,'tall':135,'team':136,'teams':137,'tetrinet':138,'the':139,'thing':140,'through':141,'time':142,'to':143,'trust':144,'unusual':145,'used':146,'using':147,'various':148,'was':149,'was ':150,'watched':151,'what':152,'what ':153,'when':154,'where':155,'where ':156,'which':157,'who':158,'who ':159,'why':160,'wide':161,'will':162,'woman':163,'worked':164,'would':165,'year':166,'you':167}
        vectorizer = CountVectorizer(vocabulary=vocab, tokenizer = mytokenizer)#, stop_words='english')
        
        X = vectorizer.fit_transform(plots)

        valid_feats = np.where(np.sum(X,0)> 2)[1]
        X = X[:,valid_feats]

#         Split Dataset into Train, Val, Test
        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
        train_ground, val_ground, test_ground,\
        train_plots, val_plots, test_plots = split_data(X, plots, labels)

        #Prune Feature Space
        common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix)
#         print('common_idx',len(common_idx))
#         return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix[:,common_idx],             np.array(train_ground), np.array(val_ground), np.array(test_ground),train_plots, val_plots, test_plots

        return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix[:,common_idx], \
                np.array(train_ground), np.array(val_ground), np.array(test_ground), vectorizer, valid_feats, common_idx, \
            train_plots, val_plots, test_plots


