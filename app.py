import streamlit as st

import requests as re
import matplotlib.pyplot as plt

import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import re
import pandas as pd
import plotly.express as px

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = joblib.load("./Vector/vectorizer.joblib")
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = RegexpTokenizer(r'[A-Za-z]+')
if 'stemmer' not in st.session_state:
    st.session_state.stemmer = SnowballStemmer("english")
if 'legitimate_phishing_df' not in st.session_state:
    Phishing_df_csv = './Dataset_Files/verified_online.csv'
    legitimate_df_csv = './Dataset_Files/top-1m.csv'

    Phishing_df = pd.read_csv(Phishing_df_csv)
    tranco_list_data = pd.read_csv(legitimate_df_csv, names=['No', 'url'])

    Phishing_df['final_target'] = 1
    tranco_list_data['final_target'] = 0

    legitimate_phishing_df = pd.concat([Phishing_df[['url', 'final_target']], tranco_list_data[['url', 'final_target']]], axis=0)
    legitimate_phishing_df.rename(columns={'url': 'legitimate_and_phishing_url'}, inplace=True)

    st.session_state.legitimate_phishing_df = legitimate_phishing_df

# Model paths
rf_model_path = r'./Models/rf_model.joblib'
xgb_model_path = r'./Models/XGB_model.joblib'
svm_model_path = r'./Models/svm_model.joblib'
dt_model_path = r'./Models/decision_model.joblib'
adaboost_model_path = r'./Models/ada_model.joblib'
nn_model_path = r'./Models/Neural_Network_model.joblib'
knn_model_path = r'./Models/knn_model.joblib'

# UI
st.markdown("<h1 style='text-align: center; font-size: 64px;'>SAFEWEB</h1>", unsafe_allow_html=True)

st.title('Phishing Website Detection using Machine Learning')
st.write('This ML-based app is developed for educational purposes...')

with st.expander("PROJECT DETAILS"):
    st.subheader('Approach')
    st.write('I used _supervised learning_...')
    st.write('The source code and data sets are available:')
    st.write('_https://github.com/emre-kocyigit/phishing-website-detection-content-based_')

    st.subheader('Data set')
    st.write('I used _"phishtank.org"_ & _"tranco-list.eu"_ as data sources.')

    labels = ['Legitimate', 'Phishing']
    sizes = [10000, 5000]
    explode = (0.05, 0.1)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
    ax.axis('equal')
    st.title("Phishing vs Legitimate Domains")
    st.markdown("Label **1** = Phishing, **0** = Legitimate")
    st.pyplot(fig)

    st.markdown('label is 1 for phishing, 0 for legitimate')
    number = st.slider("Select row number to display", 0, 100)
    st.dataframe(st.session_state.legitimate_phishing_df.head(number))

    st.subheader('Features')
    st.write('I used only content-based features...')

    st.subheader('Results')
    df_results = pd.read_csv('table.csv')
    st.table(df_results)
    st.write('XGB --> XGBoost')
    st.write('SVM --> Support Vector Machine')
    st.write('DT --> Decision Tree')
    st.write('RF --> Random Forest')
    st.write('AB --> AdaBoost')
    st.write('NN --> Neural Network')
    st.write('KN --> K-Neighbours')

with st.expander('EXAMPLE PHISHING URLs:'):
    st.write('_http://3-8-118-28.cprapid.com/info/_')
    st.write('_https://chemiluminescenc.tqt17.com/_')
    st.write('_https://defi-ned.top/h5/#/_')
    st.caption('REMEMBER, PHISHING WEB PAGES HAVE SHORT LIFECYCLE!')

# Model selection
choice = st.selectbox("Please select your machine learning model",
                      ['XGBoost', 'Support Vector Machine', 'Decision Tree',
                       'Random Forest', 'AdaBoost', 'Neural Network', 'K-Neighbours'])

if choice == 'XGBoost':
    st.session_state.model = joblib.load(open(xgb_model_path, 'rb'))
    st.write('XGB model is selected!')
elif choice == 'Support Vector Machine':
    st.session_state.model = joblib.load(open(svm_model_path, 'rb'))
    st.write('SVM model is selected!')
elif choice == 'Decision Tree':
    st.session_state.model = joblib.load(open(dt_model_path, 'rb'))
    st.write('Decision Tree model is selected!')
elif choice == 'Random Forest':
    st.session_state.model = joblib.load(open(rf_model_path, 'rb'))
    st.write('RF model is selected!')
elif choice == 'AdaBoost':
    st.session_state.model = joblib.load(open(adaboost_model_path, 'rb'))
    st.write('AB model is selected!')
elif choice == 'Neural Network':
    st.session_state.model = joblib.load(open(nn_model_path, 'rb'))
    st.write('NN model is selected!')
else:
    st.session_state.model = joblib.load(open(knn_model_path, 'rb'))
    st.write('KNN model is selected!')

# Helper Functions
def clean_url(url):
    url = url.lower()
    url = re.sub(r"https?://|www\.", "", url)
    return url

def prediction_process(url_input):
    value = clean_url(url_input)
    tokenize_value = st.session_state.tokenizer.tokenize(value)
    stem_value = [st.session_state.stemmer.stem(word) for word in tokenize_value]
    join_value = ' '.join(stem_value)
    vector = st.session_state.vectorizer.transform([join_value])
    prediction = st.session_state.model.predict(vector.toarray())
    return prediction[0]

# Input
url = st.text_input('Enter the URL')
if st.button('Check!'):
    try:
        preidcted_label_value = prediction_process(url)
        if preidcted_label_value == '0':
            st.success("This web page seems a legitimate!")
            st.balloons()
        else:
            st.warning("Attention! This web page is a potential PHISHING!")
            st.snow()
    except Exception as e:
        print("--> ", e)
