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

FLAGS = re.MULTILINE | re.DOTALL
stop_words = set(stopwords.words('english'))


print("Hola mundo")

Data = pd.read_csv('DataSet/labeled_data.csv')
'''
print(Data)
print(Data.head())
print(Data.shape)
print(Data.describe)
print(Data.dtypes)

for x in Data.columns:
    print(f"{x}\n{Data[x].unique()[:10]}")
'''

# get the number of missing data points per column
missing_values_count =Data.isnull().sum()

# look at the # of missing points in the first ten columns
print(missing_values_count[0:10])


################## PREPROCESSING: Dealing with hashtags, links and mentions. ##################
#source: https://www.pluralsight.com/guides/building-a-twitter-sentiment-analysis-in-python
#----> Hashtags
def hashtagParsing(tweet):
    return re.sub(r'\B#\w*[a-zA-Z]+\w*','<hashtag>', tweet, flags=FLAGS)

def parseMention(tweet):
    return re.sub('@[\w\-]+', '<user>', tweet)

def parseLink(tweet):
    return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','<link>', tweet, flags=FLAGS)

def removeExcessiveWhiteSpaces(tweet):
    return re.sub('\s+', ' ', tweet)

def doStemming(tweet):
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet_tokens = word_tokenize(tweet)
    words_filtered = [w for w in tweet_tokens if not w in stop_words]
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in words_filtered]
    return " ".join(stemmed_words)

def doLemmatization(tweet):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w, pos='a') for w in getTokenizedWords(tweet)]
    return " ".join(lemmatized_words)

def doLetterCasing(tweet):
    return tweet.lower()

def tokenizeWithLemmatization(tweet):
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet_tokens = word_tokenize(tweet)
    words_filtered = [w for w in tweet_tokens if not w in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w, pos='a') for w in words_filtered]
    return " ".join(lemmatized_words)

def preprocess(tweet):

    tweet = tweet.apply(removeExcessiveWhiteSpaces)
    tweet = tweet.apply(doLetterCasing)
    tweet = tweet.apply(hashtagParsing)
    tweet = tweet.apply(parseMention)
    tweet = tweet.apply(parseLink)
    #tweet = tweet.apply(tokenizeWithLemmatization)
    tweet = tweet.apply(doStemming)
    return tweet


Data['tweet'] = preprocess(Data['tweet'])
Data = Data.rename(columns={'class': 'type'})
#print(Data['tweet'])

# -- End preprocessing --

# SPLITTING DATA IN TRAINING AND TEST
testData = Data[11000:].copy()
trainingData = Data[:11000].copy()

# METHOD 1: SVN
X_train, X_test_svm, Y_train, Y_test_svm = train_test_split(Data.tweet, Data.type, test_size=0.2)

vectorizer=CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)

train_data_features=vectorizer.fit_transform(X_train)
train_data_features=train_data_features.toarray()

test_data_features=vectorizer.transform(X_test_svm)
test_data_features=test_data_features.toarray()

#SVM with linear kernel
'''
clf=svm.SVC(kernel='linear',C=1.0)
print ("Training")
clf.fit(train_data_features,Y_train)

print ("Testing")
predicted=clf.predict(test_data_features)
accuracy=np.mean(predicted==Y_test_svm)
print ("Accuracy: ",accuracy)

'''
# METHOD 2: RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

rf = RandomForestClassifier(n_estimators=400)
rf.fit(train_data_features,Y_train)
predicted=rf.predict(test_data_features)
#get_metrics(X_train, Y_train, X_test_svm, Y_test_svm, y_pr_rf_tr, y_pr_rf_val, rf)
print(accuracy_score(Y_test_svm, predicted))
print(confusion_matrix(Y_test_svm, predicted))

'''
make_confusion_matrix(cf = confusion_matrix(Y_test_svm, y_pr_rf_val),
                      X = X_test_svm,
                      y = Y_test_svm,
                      model = rf,
                      cmap='Oranges',
                      title='Confusion Matrix for Random Forest')

'''