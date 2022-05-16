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
import gensim
from gensim.models import Word2Vec


from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support



def trainLSTM(Data):

    X_train, X_test, Y_train, Y_test = train_test_split(Data.tweet, Data.type, test_size=0.3)

    # Bag of words

    #vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

    # TF-IDF

    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    



    train_data_features = vectorizer.fit_transform(X_train)
    train_data_features = train_data_features.toarray()

    test_data_features = vectorizer.transform(X_test)
    test_data_features = test_data_features.toarray()


    max_features = 10000
    lstm_out = 80

    early_stopping = callbacks.EarlyStopping(
        # monitor='accuracy',
        monitor='binary_accuracy',
        min_delta=0.005,
        patience=2,
        restore_best_weights=True,
    )
    print(train_data_features.shape)

    model = keras.Sequential([
        layers.Embedding(max_features, 64),
        layers.LSTM(units=lstm_out),
        layers.Dropout(0.3),
       # layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])


    #model.compile(optimizer='rmsprop',
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])

    history = model.fit(train_data_features, Y_train,

                        epochs=2,
                        batch_size=32,
                        #validation_split=0.2,
                        validation_data=(test_data_features, Y_test),
                        callbacks=[early_stopping])



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
