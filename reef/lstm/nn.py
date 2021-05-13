import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepNet, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.linear_1(x))
        out = F.relu(self.linear_2(out))
        return self.out(out)
#top_words=5000

def lstm_simple(train_text, y_train, test_text, y_test, bs=64, n=3, dataset='imdb'):
    #Label Processing
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
#     if dataset =='trec':
#         y_train *= 1
#     if dataset != 'audit':
        #Make Tokenizer
#         tokenizer = Tokenizer()
#         tokenizer.fit_on_texts(train_text)
#         tokenizer.fit_on_texts(test_text)
#         X_train = tokenizer.texts_to_sequences(train_text)
#         X_test = tokenizer.texts_to_sequences(test_text)
        # print('audit')

    #Make embedding 
    if dataset == 'audit':
        print(y_train)
        y_train[y_train>=0.5] =1
        y_train[y_train<0.5] =0
        print('train_text.shape', y_train)
        model = LogisticRegression(random_state=25).fit(train_text, y_train)
        scores = model.predict(train_text)
        X_test = test_text
        y_pred = model.predict(X_test)

    else:

        max_sentence_length= 768
        X_train = train_text#sequence.pad_sequences(X_train, maxlen=max_sentence_length)
        X_test = test_text#sequence.pad_sequences(X_test, maxlen=max_sentence_length)
        embedding_vector_length = 768 #32
        vocab_size=768 #len(tokenizer.word_index) + 1
        # print('before ',  X_train.shape)
        # X_train = X_train.reshape(-1, 768, 1)
        # X_test = X_test.reshape(-1, 768, 1)
        # print('after',  X_train.shape)

        X_train  = torch.tensor(X_train)
        X_test  = torch.tensor(X_test)
        y_train  = torch.tensor(y_train).long()
        # y_test = 
        #Model Architecture
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True,pin_memory=True)
        model = DeepNet(768, 512, 2) #n_features, n_hidden, n_classes
        optimizer_lr = torch.optim.Adam(model.parameters(), lr= 0.0003)
        # print(model.summary())
        supervised_criterion = torch.nn.CrossEntropyLoss()
        for i in range(n):
            model.train()
            # loss = 0
            for batch_ndx, sample in enumerate(loader):
                loss = supervised_criterion(model(sample[0]), sample[1])
                loss.backward()
                optimizer_lr.step()
            print('Loss ', loss)


        probs = torch.nn.Softmax()(model(X_test))
        y_pred = np.argmax(probs.cpu().detach().numpy(), 1)
            
        
        # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=n, batch_size=bs)
        
        
    return y_pred
