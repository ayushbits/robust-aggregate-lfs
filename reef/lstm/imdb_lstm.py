import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression
#top_words=5000

def lstm_simple(train_text, y_train, test_text, y_test, bs=64, n=3, dataset='imdb'):
    #Label Processing
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
#     if dataset =='trec':
#         y_train *= 1
    if dataset != 'audit':
        #Make Tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_text)
        tokenizer.fit_on_texts(test_text)
        X_train = tokenizer.texts_to_sequences(train_text)
        X_test = tokenizer.texts_to_sequences(test_text)
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

        max_sentence_length=500
        X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
        X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)
        embedding_vector_length = 32
        vocab_size=len(tokenizer.word_index) + 1
        print('vocab_size is' , vocab_size)

        #Model Architecture
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_sentence_length))
        model.add(LSTM(100))
        if dataset=='trec':
            model.add(Dense(6, activation="softmax"))
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            model.add(Dense(1, activation="sigmoid"))
            #Run the model!
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        # print(X_train.shape, y_train.shape)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=n, batch_size=bs)
        if dataset != 'trec':
            scores = model.evaluate(X_test, y_test, verbose=0)
            print("Accuracy: %.2f%%" % (scores[1]*100))

        #y_pred = model.predict(X_test, batch_size=1)
        if dataset == 'trec':
            y_pred = model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1)

        else:
            y_pred = model.predict(X_test, batch_size=1)
            print("y_pred is", y_pred)
            y_pred = np.array([x[0] for x in y_pred])
    return y_pred
