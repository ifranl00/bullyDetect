import numpy
import pandas as pd
from LSTM import trainLSTM
from LSTM_w2v import trainLSTM_w2v
from RandomForest import trainRandomForest
from SVN import trainSVN
from sklearn.feature_extraction.text import CountVectorizer
import gensim.models.keyedvectors as word2vec #need to use due to depreceated model
from keras.preprocessing.text import Tokenizer
from Preprocessing import preprocess, preprocessInp
import streamlit as st
import pickle
import numpy as np
from keras.models import load_model
from Preprocessing import preprocess
import pickle
from keras.preprocessing.sequence import pad_sequences

# -- Preprocessing of datasets --





# -- Web App --
# source: https://www.youtube.com/watch?v=xl0N7tHiwlw


def train():
    Data = pd.read_csv('DataSet/labeled_data.csv')
    Data['tweet'] = preprocess(Data['tweet'])
    Data = Data.rename(columns={'class': 'type'})
    Data['type'] = Data['type'].map({0: 1, 1: 1, 2: 0})
    #trainRandomForest(Data)
    #trainSVN(Data)
    #trainLSTM(Data)
    #trainLSTM_w2v(Data)

def show_predict_page():
    st.title("Software Developer Bullying Prediction")
    st.write("""### Please write the message to evaluate as bullying or not""")
    #https://discuss.streamlit.io/t/how-to-take-text-input-from-a-user/187
    message_input = st.text_area("Message to analyze","Enter message")

    #mejora: hacer una encuesta o despues de si piensa que es bullying o no y esto se anyade a datos?
    ok = st.button("Detect bullying")
    if ok:
        #preproccessing
        model = load_model('W2v_model.h5')
        message_input = preprocessInp(message_input)


        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        listInput = []
        listInput.append(message_input)
        X = tokenizer.texts_to_sequences(listInput)
        # lenght of tweet to consider
        maxlentweet = 10
        # add padding


        X = pad_sequences(X, maxlen=maxlentweet)



        prediction = model.predict(X)
        print(prediction)


        porcentaje = prediction[0][0] * 100


        if (porcentaje < 50):
            porcentaje = 100 - porcentaje

            st.success(f"NOT bully, with an  {(round(float(porcentaje), 1))} %")

        else:
            st.error(f"BULLY, with an {(round(float(porcentaje), 1))} %")



#show_predict_page()

train()
