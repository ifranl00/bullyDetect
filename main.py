import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import matplotlib.pyplot as plt
import re
import string

from LSTM import trainLSTM
from RandomForest import trainRandomForest
from SVN import trainSVN
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
def hashtagParsing(tweet): #TODO: just delete # for detecting at least when there are composed by one word and split the ones that are hiHowAre form.
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
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet_tokens = word_tokenize(tweet)
    words_filtered = [w for w in tweet_tokens if not w in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w, pos='a') for w in words_filtered]
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
Data['type']= Data['type'].map({0: 1, 1: 1, 2: 0})
#print(Data['tweet'])

# -- End preprocessing --

# -- Init trainning --

#trainRandomForest(Data)
#trainSVN(Data)
trainLSTM(Data)




