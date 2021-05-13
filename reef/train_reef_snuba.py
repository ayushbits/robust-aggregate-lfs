#python train_reef_snuba.py imdb normal val_5_dict_dt1
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer
# from lstm.imdb_lstm import *
# from lstm.lstm import *
# from lstm.nn import *
from lstm.DeepLSTM import *
import warnings
import sys, os
warnings.filterwarnings('ignore')

dataset= sys.argv[1]
print('dataset is ', dataset)
loader_file = "data." + dataset+"_loader"

import importlib

load = importlib.import_module(loader_file)

dl = load.DataLoader()
train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, train_ground,\
    val_ground, test_ground, _,_,_, train_text, val_text, test_text = dl.load_data(dataset=dataset, split_val = 0.1)

    
    
mode = sys.argv[2]
save_dir = sys.argv[3]
pickle_save = "LFs/"+ dataset + "/" + save_dir

file_name = mode + '_reef.npy'
train_reef = np.load(os.path.join(pickle_save,file_name))
# configuration = DistilBertConfig()
# model_class = DistilBertModel(configuration)
# tokenizer_class, pretrained_weights = DistilBertTokenizer, 'distilbert-base-uncased'

# tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
# model = model_class.from_pretrained(pretrained_weights)

# df_train = pd.DataFrame(train_text)
# df_val = pd.DataFrame(val_text)
# df_test = pd.DataFrame(test_text)

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
print("X_train shape ", X_train.shape)

f1_all = []
pr_all = []
re_all = []
test_acc_all = []


# bs_arr = [64]#,128,256]
bs = 64
epochs = 15
train_ground[train_ground == -1] =0
test_ground[test_ground == -1] = 0
print('vocab_size, embedding_vector_length, max_sentence_length',vocab_size, embedding_vector_length, max_sentence_length)
# for n in range(epo:
y_pred = lstm_simple(X_train, train_ground,	X_test, test_ground, vocab_size, embedding_vector_length, max_sentence_length, bs, epochs)

predictions = np.round(y_pred)

test_acc_all.append(np.sum(predictions == test_ground)/float(np.shape(test_ground)[0]))
f1 = f1_score(test_ground, predictions, average='macro')

f1_all.append(f1)
pr_all.append(precision_score(test_ground, predictions))
re_all.append(recall_score(test_ground, predictions))
    # if n == n_epochs_arr[0] -1:
print('Accuracy ',test_acc_all, 'F1 score', f1_all, 'Precision', pr_all, 'Recall',re_all)

