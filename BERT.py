import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import matplotlib.pyplot as plt
import re
import string
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import callbacks

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support



def train(Data):
    X_train, X_test, Y_train, Y_test = train_test_split(Data.tweet, Data.type, test_size=0.3)
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    train_data_features = vectorizer.fit_transform(X_train)
    train_data_features = train_data_features.toarray()

    test_data_features = vectorizer.transform(X_test)
    test_data_features = test_data_features.toarray()

    max_features = 10000

    early_stopping = callbacks.EarlyStopping(
        # monitor='accuracy',
        monitor='acc',
        min_delta=0.005,
        patience=2,
        restore_best_weights=True,
    )
    print(train_data_features.shape)

    model = keras.Sequential([
        layers.Embedding(max_features, 64),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
       # layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])


    #model.compile(optimizer='rmsprop',
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(train_data_features, Y_train,

                        epochs=5,
                        batch_size=128,
                        validation_split=0.2,
                        callbacks=[early_stopping])

    print(history.history)

    plt.plot(history.history['acc'])
    plt.show()
    plt.plot(history.history['loss'])
    plt.show()

    history_df = pd.DataFrame(history.history)
    # Start the plot at epoch 5
    history_df.loc[5:, ['loss', 'val_loss']].plot()
    history_df.loc[5:, ['acc', 'val_acc']].plot()

    print(("Best Validation Loss: {:0.4f}" + \
           "\nBest Validation Accuracy: {:0.4f}") \
          .format(history_df['val_loss'].min(),
                  history_df['val_acc'].max()))

    score = model.evaluate(test_data_features, Y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
