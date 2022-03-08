import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import matplotlib.pyplot as plt
import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support

def trainSVN(Data):
    X_train, X_test_svm, Y_train, Y_test_svm = train_test_split(Data.tweet, Data.type, test_size=0.3)

    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

    train_data_features = vectorizer.fit_transform(X_train)
    train_data_features = train_data_features.toarray()

    test_data_features = vectorizer.transform(X_test_svm)
    test_data_features = test_data_features.toarray()

    # SVM with linear kernel
    # '''
    clf = svm.SVC(kernel='linear', C=1.0)
    print("Training")
    clf.fit(train_data_features, Y_train)

    print("Testing")
    predicted = clf.predict(test_data_features)
    accuracy = np.mean(predicted == Y_test_svm)
    print("Accuracy: ", accuracy)