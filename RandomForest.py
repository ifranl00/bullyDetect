from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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

def trainRandomForest(Data):
    X_train, X_test_svm, Y_train, Y_test_svm = train_test_split(Data.tweet, Data.type, test_size=0.3)

    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

    train_data_features = vectorizer.fit_transform(X_train)
    train_data_features = train_data_features.toarray()

    test_data_features = vectorizer.transform(X_test_svm)
    test_data_features = test_data_features.toarray()


    rf = RandomForestClassifier(n_estimators=400, random_state=1)
    rf.fit(train_data_features, Y_train)
    predicted = rf.predict(test_data_features)
    # get_metrics(X_train, Y_train, X_test_svm, Y_test_svm, y_pr_rf_tr, y_pr_rf_val, rf)
    print(accuracy_score(Y_test_svm, predicted))
    print(confusion_matrix(Y_test_svm, predicted))