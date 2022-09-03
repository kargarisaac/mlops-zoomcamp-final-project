import streamlit as st
import requests

st.set_page_config(
    page_title="Fashion Sentiment Analysis",
    page_icon="📈",
    layout="wide",
)

st.header("Fashion Sentiment Analysis UI")

#############################################################################################
st.image("https://krm-stc-ms.azureedge.net/-/media/Images/Ecco/Products/MENS/BROWN/ECCO-SOFT-7-M/470364-02053-main.webp?Crop=1&Size=ProductDetailsMedium1x")
review_text = st.text_input('User review about the product:', 'absolut wonder silki sexi comfort.')

## map
if st.button("Analyze"):
    endpoint = "https://sentiment-lr4ixkffqq-ew.a.run.app"
    # endpoint = "http://localhost:8080"
    url = f'{endpoint}/predict?review={review_text}'
    
    prediction = requests.get(url).json()
    st.write("Prediction:", prediction)
