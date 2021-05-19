#python train_reef_snuba.py imdb (1) normal (2) val_5_dict_dt1 (3) lstm/nn (4) num_epochs (5)
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer
# from lstm.imdb_lstm import *
# from lstm.lstm import *
# from lstm.nn import *
from lstm.DeepLSTM import *
import warnings
import sys, os
warnings.filterwarnings('ignore')
import pickle
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
print(device)


dataset= sys.argv[1]
print('dataset is ', dataset)
model = sys.argv[4]
mode = sys.argv[2]
save_dir = sys.argv[3]
pickle_save = "LFs/"+ dataset + "/" + save_dir

file_name = mode + '_reef.npy'
train_reef = np.load(os.path.join(pickle_save,file_name))

if model == "lstm":
	loader_file = "data." + dataset+"_loader"

	import importlib

	load = importlib.import_module(loader_file)

	dl = load.DataLoader()
	train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, train_ground,\
	    val_ground, test_ground, _,_,_, train_text, val_text, test_text = dl.load_data(dataset=dataset, split_val = 0.1)
	train_ground = train_reef

elif model == "nn" or model == "feat_lstm":
	objs = []
	fname = pickle_save + "/" + mode + "_d_processed.p"
	with open(fname, 'rb') as f:
	    while 1:
	        try:
	            o = pickle.load(f)
	        except EOFError:
	            break
	        objs.append(o)

	x_supervised = torch.tensor(objs[0]).double()
	y_supervised = torch.tensor(objs[3]).long()
	print('supervised shape', objs[2].shape)

	objs = []
	if mode != '':
	    fname = pickle_save + "/" + mode + "_U_processed.p"
	else:
	    fname = pickle_save + "/U_processed.p"

	with open(fname, 'rb') as f:
	    while 1:
	        try:
	            o = pickle.load(f)
	        except EOFError:
	            break
	        objs.append(o)


	x_unsupervised = torch.tensor(objs[0]).double()
	y_unsupervised = torch.tensor(objs[3]).long()
	print('UNsupervised shape', objs[2].shape)
	print('Length of U is', len(x_unsupervised))

	objs = []
	if mode != '':
	    fname = pickle_save + "/" + mode + "_validation_processed.p"
	else:
	    fname = pickle_save + "/validation_processed.p"

	with open(fname, 'rb') as f:
	    while 1:
	        try:
	            o = pickle.load(f)
	        except EOFError:
	            break
	        objs.append(o)

	x_valid = torch.tensor(objs[0]).double()
	y_valid = objs[3]
	print('Valid shape', x_valid.shape)
	objs1 = []
	if mode != '':
	    fname = pickle_save + "/" + mode + "_test_processed.p"
	else:
	    fname = pickle_save + "/test_processed.p"


	with open(fname, 'rb') as f:
	    while 1:
	        try:
	            o = pickle.load(f)
	        except EOFError:
	            break
	        objs1.append(o)
	x_test = torch.tensor(objs1[0]).double()
	y_test = objs1[3]
	print('Test shape', x_test.shape)
	n_features = x_supervised.shape[1]

	X_train = x_unsupervised
	train_ground = train_reef
	X_test = x_test
	test_ground = y_test
    


# configuration = DistilBertConfig()
# model_class = DistilBertModel(configuration)
# tokenizer_class, pretrained_weights = DistilBertTokenizer, 'distilbert-base-uncased'

# tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
# model = model_class.from_pretrained(pretrained_weights)

# df_train = pd.DataFrame(train_text)
# df_val = pd.DataFrame(val_text)
# df_test = pd.DataFrame(test_text)
if model == "lstm":
	def get_features(text):
		tokenized = text[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
		max_len = 0
		for i in tokenized.values:
		    if len(i) > max_len:
		        max_len = len(i)

		padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

		np.array(padded).shape

		attention_mask = np.where(padded != 0, 1, 0)
		attention_mask.shape

		input_ids = torch.tensor(padded)  
		attention_mask = torch.tensor(attention_mask)

		with torch.no_grad():
		    last_hidden_states = model(input_ids, attention_mask=attention_mask)

		train_features = last_hidden_states[0][:,0,:].numpy()

		return train_features

	# X_train = get_features(df_train) 
	# # X_train = train_text
	# # X_val = get_features(df_val)
	# X_test = get_features(df_test) #test_text
	# # X_test = test_text

	mkt = MakeTokens()
	X_train, X_val, X_test, vocab_size, embedding_vector_length, max_sentence_length =\
	 mkt.make(train_text, val_text, test_text)
	# print("X_train shape ", X_train.shape)

	print('vocab_size, embedding_vector_length, max_sentence_length',vocab_size, embedding_vector_length, max_sentence_length)
	# for n in range(epo:
	n_features  = X_train.shape[1]

print('X_train.shape, len(train_ground)', X_train.shape, len(train_ground))
print('X_test.shape, len(test_ground)', X_test.shape, len(test_ground))

f1_all = []
pr_all = []
re_all = []
test_acc_all = []


# bs_arr = [64]#,128,256]
bs = 64
epochs = int(sys.argv[5])
train_ground[train_ground == -1] =0
test_ground[test_ground == -1] = 0
y_pred = np.zeros(len(test_ground))

X_train  = torch.tensor(X_train).long().to(device)
X_test  = torch.tensor(X_test).long().to(device)
train_ground  = torch.tensor(train_ground).float().to(device)
# print(train_ground.shape)
# train_ground = train_ground.reshape(-1)
# print(train_ground.shape)
# test_ground  = torch.tensor(test_ground).float().to(device)

# print('train reef ', train_ground[0:10])
num_runs =3
for i in range(0,num_runs):
	if model == 'lstm':
		y_pred = lstm_simple(model, X_train, train_ground, X_test, \
			test_ground, vocab_size, embedding_vector_length, max_sentence_length, bs, epochs)
	elif model == 'nn':
		y_pred = lstm_simple(model, X_train, train_ground,	X_test, test_ground, bs =  bs, \
			epochs = epochs, num_feats = n_features)
	elif model == 'feat_lstm':
		lstm_simple(model, X_train, train_ground, X_test, test_ground, vocab_size = n_features, \
			max_sentence_length= n_features, bs =  bs, epochs = epochs, num_feats = n_features)

	predictions = np.round(y_pred)
	# print(type(predictions))
	predictions = predictions
	# test_acc_all = np.sum(predictions == test_ground)/float(np.shape(test_ground)[0])
	acc = accuracy_score(test_ground, predictions)
	f1 = f1_score(test_ground, predictions, average='macro')

	f1_all.append(f1)
	pr_all.append(precision_score(test_ground, predictions, average='macro'))
	re_all.append(recall_score(test_ground, predictions, average='macro'))
    # if n == n_epochs_arr[0] -1:
print('F1 score', np.mean(f1_all), np.std(f1_all)) 
print('precisiom score', np.mean(pr_all), np.std(pr_all))
print('recall score', np.mean(re_all), np.std(re_all))

