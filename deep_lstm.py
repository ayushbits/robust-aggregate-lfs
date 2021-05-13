import torch.nn as nn
import torch.nn.functional as F
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.data import TensorDataset, DataLoader        

import numpy as np

class MakeTokens():
    def __init__(self):
        self.max_sentence_length = 500
        self.embedding_vector_length = 32

    def make(self,train_text, val_text, test_text):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_text)
        tokenizer.fit_on_texts(val_text)
        tokenizer.fit_on_texts(test_text)
        X_train = tokenizer.texts_to_sequences(train_text)
        X_val = tokenizer.texts_to_sequences(val_text)
        X_test = tokenizer.texts_to_sequences(test_text)

        X_train = sequence.pad_sequences(X_train, maxlen=self.max_sentence_length)
        X_val = sequence.pad_sequences(X_val, maxlen=self.max_sentence_length)
        X_test = sequence.pad_sequences(X_test, maxlen=self.max_sentence_length)
        vocab_size=len(tokenizer.word_index) + 1
        return X_train, X_val, X_test, vocab_size, self.embedding_vector_length, self.max_sentence_length



class DeepLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_vector_length, max_sentence_length, num_classes=1):
        super(DeepLSTM, self).__init__()
        self.hidden_size = 100
        self.embedding_vector_length = embedding_vector_length
        self.max_sentence_length = max_sentence_length
        self.emb = nn.Embedding(vocab_size, embedding_dim = self.embedding_vector_length)
        self.lstm1 = nn.LSTM(input_size = embedding_vector_length, hidden_size = self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        emb = self.emb(x)
        # print('emb.shape', emb.shape)
        # x = emb.trans pose(0,1)
        x,_ = self.lstm1(emb)
        x = x[:,-1,:]
        # print('x.shape', x.shape)
        # x = x.view(-1, 50000)
        x = self.out(x)
        x = self.sig(x)
        return x

    def get_embedding(self, x):
        return self.emb(x).view(-1, self.max_sentence_length*self.embedding_vector_length)
