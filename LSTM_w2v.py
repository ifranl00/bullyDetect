import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import matplotlib.pyplot as plt
import re
import string
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import callbacks
import seaborn as sns
import gensim.models.keyedvectors as word2vec #need to use due to depreceated model
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support , roc_curve,  roc_auc_score
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer



def trainLSTM_w2v(Data):

    tweets = Data['tweet']
    labels = Data['type']

    tweets_split = []

    for i, line in enumerate(tweets):

        tweets_split.append(line)

    print(tweets_split[1])

    w2vModel = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True,
                                                          limit=50000)



    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets_split)
    X = tokenizer.texts_to_sequences(tweets_split)

    # lenght of tweet to consider
    maxlentweet = 10
    # add padding
    X = pad_sequences(X, maxlen=maxlentweet)
    print(X.shape)

    # create a embedding layer using Google pre triained word2vec (50000 words)
    embedding_layer = Embedding(input_dim=w2vModel.syn0.shape[0], output_dim=w2vModel.syn0.shape[1],
                                weights=[w2vModel.syn0],
                                input_length=X.shape[1])

    # create model

    lstm_out = 80





    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.3, random_state=24)





    max_features = 10000

    early_stopping = callbacks.EarlyStopping(
        # monitor='accuracy',
        monitor='binary_accuracy',
        min_delta=0.005,
        patience=5,
        restore_best_weights=True,
    )

    model = keras.Sequential([
        embedding_layer,
        layers.LSTM(units=lstm_out),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])
    print(model.summary())

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])


    history = model.fit(X_train, Y_train,

                        epochs=10,
                        batch_size=32,
                        #validation_split=0.2,
                        validation_data=(X_test, Y_test),
                        callbacks=[early_stopping])

    '''
    # ROC AUC curve
    rocAuc = roc_auc_score(Y_test, y_pred)

    falsePositiveRate, truePositiveRate, _ = roc_curve(Y_test, y_pred)

    print("ROC")
    print(falsePositiveRate,truePositiveRate )
    '''

    history_df = pd.DataFrame(history.history)
    history_df = history_df.rename(columns={'binary_accuracy': 'accuracy'})
    history_df = history_df.rename(columns={'val_binary_accuracy': 'val_accuracy'})

    plt.title("Training and validation loss results")
    sns.lineplot(data=history_df['loss'], label="Training Loss")
    sns.lineplot(data=history_df['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    plt.title("Training and validation accuracy results")
    sns.lineplot(data=history_df['accuracy'], label="Training Accuracy")
    sns.lineplot(data=history_df['val_accuracy'], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

    print(history_df.iloc[-1])
