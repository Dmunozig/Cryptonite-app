import streamlit as st
import time
import datetime
import pandas as pd
import numpy as np
from PIL import Image
import requests

# page conf
st.set_page_config(
    page_title="Cryptonite",
    page_icon=":gem:",
    layout="centered", # wide
    initial_sidebar_state="expanded") # collapsed

# Page coloring
CSS = """
h1 {
    color: green;
}
body {
    color: black;
}
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

# Background image

import base64

@st.cache
def load_image(path):
    with open(path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    return encoded

def image_tag(path):
    encoded = load_image(path)
    tag = f'<img src="data:image/png;base64,{encoded}">'
    return tag

def background_image_style(path):
    encoded = load_image(path)
    style = f'''
    <style>
    body {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: pattern;
    }}
    </style>
    '''
    return style

image_path = 'images/congruent_pentagon_green_background.png'
st.write(background_image_style(image_path), unsafe_allow_html=True)


# st.sidebar


# Logo and Title 

'''
# Cryptonite... 
'''

col1, col2, col3 = st.beta_columns(3)
image = Image.open('images/bit-emerald.png')

with col2:
    st.image(image)

# st.image(image, width=None)
'''
# ...*Sentiment-Based Bitcoin Predictions*
#
'''
'''
### Bitcoin price, unlike stocks or bonds, is **not tied to any tangible asset**. Could we assume **people's opinion** of the cryptocurrency is it's main driving force?
'''

'''
### We set out to predict the fluctuations of the bitcoin market using machine learning models and then quantify the effects of sentiment indicators.
At first we tried to develop our own sentiment analysis tool, but we found pre-exisiting cryptocurrency sources giving sentiment analysis of social media:

1. [Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/) : This index takes into account not only social media sentiment (accounts for 15% of overall value) but also other economic indicators like Volatility (25%) or Market Momentum (25%). This gives an added economical inight.
2. [Augmento Sentiment Data](https://www.augmento.ai/) : This source held a count of labelled messages from different sources (twitter, reddit and bitcoin talk) which we then filtered for positive, negative and neutral emotions. More on methodology below.

By using these indexes we allowed ourselves to focus on our time-series predictive model. Ultimately we used Facebook's Prophet model due to its predictive capacity, seasonality fits, and ease of use.
Below you will find contrasting results for predictions with and without the use of sentiment indicators...

'''

### params for our potential API
# params = dict(
#         start_datetime=[f"{start_datetime} UTC"],
#         end_datetime=[f"{end_datetime} UTC"],
#         wallet=[float(wallet)])


## EXAMPLE API REQUEST, TO BE DELETED ONCE OURS IS UP AND RUNNING

# import datetime
# d = st.date_input('Ride Date', datetime.datetime(2020, 4, 20))
# t = st.time_input('Ride Time', datetime.time(16, 20))
# datetime = datetime.datetime.combine(d, t)


# p_lon = st.number_input('Insert pick-up longitude')
# p_lat = st.number_input('Insert pick-up latitude')
# d_lon = st.number_input('Insert dropoff longitude')
# d_lat = st.number_input('Insert dropoff latitude')


# count = st.number_input('Insert passenger count')

# url = 'http://taxifare.lewagon.ai/predict_fare/'

# params = dict(
#         key=["2013-07-06 17:18:00.000000119"],
#         pickup_datetime=[f"{datetime} UTC"],
#         pickup_longitude=[float(p_lon)],
#         pickup_latitude=[float(p_lat)],
#         dropoff_longitude=[float(d_lon)],
#         dropoff_latitude=[float(d_lat)],
#         passenger_count=[int(count)])

# import requests
# response = requests.get(url, params=params)


# prediction = response.json()

# st.write('...based on previous price fluctuations:', round(prediction['prediction'],2), '$')


'''
# 
# Example investment wallets

### These wallets will allow you to explore the effects of the analyzed sentiment indexes on your potential investments over time if they were done in the past.
###
Please choose a time frame and wallet budget (USD $):
'''
buget = st.number_input('Insert initial investment (USD $)', value = 100, min_value = 100, max_value = 10_000_000)
d_start = st.date_input('Start of investment', datetime.datetime(2020, 4, 20), min_value=datetime.datetime(2019, 6, 14), max_value=datetime.datetime(2021, 3, 1))
d_end = st.date_input('End of investment', datetime.datetime(2020, 4, 20), min_value=datetime.datetime(2019, 6, 14), max_value=datetime.datetime(2021, 3, 1))

# Example graph showcasing trends of wallet with and without the indexes
@st.cache
def get_line_chart_data():
    return pd.DataFrame(
            np.random.randn(20, 3),
            columns=['a', 'b', 'c']
        )

# We are gonna load csv with the dataframe in reality
df = get_line_chart_data()

# convert the user date input into relevant indexes? Yassine's function might do that 
st.line_chart(df)

'''
### Your wallet gains are...
'''

st.write('...based on previous price fluctuations:', round(3), '$')
st.write('...based on previous price and ONE sentiment index indicator:', round(5), '$')
st.write('...based on previous price and TWO sentiment index indicators:', round(7), '$')


'''
# 
# Exploratory Data Analysis and Metrics
#

Prophet uses a decomposable time series model with three main model components: trend, seasonality, and holidays. They are combined in the following equation:
'''
# For mathematical functions use latex
st.latex(r'''
    y(t)= g(t) + s(t) + h(t) + εt
    ''')
'''
> * g(t): piecewise linear or logistic growth curve for modeling non-periodic changes in time series
> * s(t): periodic changes (e.g. weekly/yearly seasonality)
> * h(t): effects of holidays (user provided) with irregular schedules
> * εt: error term accounts for any unusual changes not accommodated by the model
'''
'''
Our model does not pick up the latest spike (Nov/20 to Feb/21), pointing towards the probable hypothesis 
Obviously no model will ever beat having predicted the bigges (outlier) value spike in the history of the crypto market!
'''

'''
# 
# Live predictor indicator v1.0:
#
Our API is based on both the 12:00AM UTC BTC closing value and the Fear&Greed index value that is updated at the same time.
For this reason our API will be updated every day at 12:05AM

## Tomorrow's prediction is:
'''

# Need to change url once it is changed to docker
@st.cache
def response():
    url = 'http://127.0.0.1:8000'
    return requests.get(url)

prediction = response().json()

st.write('Predicted BTC Price:', round(prediction['prediction'],2), '$')
st.write('Closing BTC Price:', round(prediction['current_btc_price'],2), '$')
st.write('Predicted change:', round(100-((prediction['current_btc_price']/prediction['prediction'])*100),2), '%')
st.write('## Buy/Sell Recommendation:', prediction['Recommendation'], '!')
if prediction['Recommendation'] == 'Sell':
    st.error('You should sell today, the sooner the better!')
if prediction['Recommendation'] == 'Buy':
    st.success('You should buy bitcoin ASAP!')

# Timer until next update

# Graph showing evolution of F&G progress (we have the df so better to plot it ourselves)
