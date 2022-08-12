import streamlit as st
import requests

st.set_page_config(
    page_title="Fashion Sentiment Analysis",
    page_icon="📈",
    layout="wide",
)

#############################################################################################
review_text = st.text_input('User review', 'absolut wonder silki sexi comfort')

method = st.selectbox(
     'Method',
     ('TFIDF', 'BoW', 'Deep Learning'))

methods = {
    'TFIDF': 'tfidf',
    'BoW': 'bow',
    'Deep Learning': 'dl'
}
## map
if st.button("Analyze"):
    url = f'http://localhost:8000/predict?review={review_text}&method={methods[method]}'
    prediction = requests.get(url).json()
    st.write("Prediction:", prediction)
