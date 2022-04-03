import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from Preprocessing import preprocess, preprocessInp
import streamlit as st
import pickle
import numpy as np
from keras.models import load_model
from Preprocessing import preprocess

# -- Preprocessing of datasets --
Data = pd.read_csv('DataSet/labeled_data.csv')
Data['tweet'] = preprocess(Data['tweet'])
Data = Data.rename(columns={'class': 'type'})
Data['type']= Data['type'].map({0: 1, 1: 1, 2: 0})

# -- Init trainning --
#trainRandomForest(Data)
#trainSVN(Data)
#trainLSTM(Data)

# -- Web App --
# source: https://www.youtube.com/watch?v=xl0N7tHiwlw
model = load_model('my_model.h5')

def show_predict_page():
    st.title("Software Developer Bullying Prediction")
    st.write("""### Please write the message to evaluate as bullying or not""")
    #https://discuss.streamlit.io/t/how-to-take-text-input-from-a-user/187
    message_input = st.text_area("Message to analyze","Enter message")

    #mejora: hacer una encuesta o despues de si piensa que es bullying o no y esto se anyade a datos?
    ok = st.button("Detect bullying")
    if ok:
        #preproccessing
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(message_input)
        message_input = preprocessInp(message_input)
        vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                     max_features=5000)
        vectorizedInput = vectorizer.transform([message_input])
        prediction = model.predict(vectorizedInput)[0]
        print(prediction)
        st.subheader(f"The estimated salary is ${prediction[0]:.2f}")


show_predict_page()






