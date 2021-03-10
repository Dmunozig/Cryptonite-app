import streamlit as st
import time
import datetime
import pandas as pd
import numpy as np
from PIL import Image
import requests
import altair as alt

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

# Sidebar
st.sidebar.markdown(f"""
    # Cryptonite Site Navigation
    """)

## page navigation
pages = st.sidebar.radio('Select page:', ('Main', 'Live Prediction', 'EDA & Metrics'))

st.sidebar.markdown(f"""
    #
    #
    ### Team:
    - Olavo Watanabe
    - Yassine Rkaibi
    - Imamul Alam
    - Diego Muñoz 
    ### 
    Thank you to the Le Wagon staff for providing us with the skill and help to develop this app.
    
    SO to Clementine Contat for guidance through the project!
    """)



if pages == 'Main':
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

    # Intro:

    '''
    ### Bitcoin price, unlike stocks or bonds, is **not tied to any tangible asset**. Could we assume **people's opinion** of the cryptocurrency is it's main driving force?
    '''

    '''
    ### We set out to predict the fluctuations of the bitcoin market using machine learning models and then quantify the effects of sentiment indicators.
    At first we tried to develop our own sentiment analysis tool, but we found pre-exisiting cryptocurrency sources giving sentiment analysis of social media:

    1. [Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/) : This index takes into account not only social media sentiment (accounts for 15% of overall value) but also other economic indicators like **Volatility** (25%) or **Market Momentum/Volume** (25%). This gives an added economical inight to our predictions.
    2. [Augmento Sentiment Data](https://www.augmento.ai/) : This source held a count of labelled messages from different sources (twitter, reddit and bitcoin talk) which we then filtered for **positive, negative** and **neutral emotions**. More on our methodology can be found in EDA and Metrics page.

    By using these indexes we freed ourselves to focus on the time-series predictive model. Ultimately we used Facebook's Prophet model due to its predictive capacity, seasonality fits, and ease of use.
    Below you will find contrasting results for predictions with and without the use of sentiment indicators...

    '''

    # Interactive investment wallets:

    '''
    # 
    # Interactive investment wallets
    **Potential wallet names:**
    - no score / one score / all scores
    - wallet 1.0 / wallet 2.0 / wallet 3.0
    - baseline / 1 index / 2 indexes
    '''
    # Load csv with predictions
    no_score = pd.read_csv('data/predictions_no_score.csv')
    one_score = pd.read_csv('data/predictions_fear_greed_score.csv')
    all_score = pd.read_csv('data/predictions_all_scores_best.csv')

    '''
    ### These wallets will allow **you** to explore the effects of the analyzed sentiment indexes on your *potential* investments over time... If they were done in the past.
    ###
    Please choose a time frame and wallet budget (USD $):
    '''
    # user inputs
    budget = st.number_input('Insert initial investment (USD $)', value = 100, min_value = 100, max_value = 10_000_000)
    d_start = st.date_input('Start of investment', datetime.datetime(2019, 6, 19), min_value=datetime.datetime(2019, 6, 19), max_value=datetime.datetime(2021, 1, 31))
    d_end = st.date_input('End of investment', datetime.datetime(2021, 1, 31), min_value=datetime.datetime(2019, 6, 19), max_value=datetime.datetime(2021, 1, 31))

    # Function for wallet/portfolio investing
    def wallet(budget, df, start='2019-06-19', end='2021-01-31'):
        start = df[df['ds']==start].index[0]
        end = df[df['ds']==end].index[0]
        new_df = df[start:end]
        cash = budget
        btc_value = 0
        market_portfolio = (budget/new_df['y'][start])*new_df['y'][end-1]
        portfolio_value = []
        transaction_dates = []
        for index, row in new_df.iterrows():
            if row['future_change']==1:
                if cash==0:
                    transaction_dates.append(row['ds'])
                    portfolio_value.append(btc_value*row['y'])
                else:
                    btc_value = (cash/row['y'])
                    transaction_dates.append(row['ds'])
                    portfolio_value.append(cash)
                    cash = 0
            if row['future_change']==0:
                if len(portfolio_value)==0:
                    portfolio_value.append(budget)
                    transaction_dates.append(row['ds'])
                else:
                    transaction_dates.append(row['ds'])
                    portfolio_value.append(portfolio_value[-1])
                    if btc_value>0:
                        cash = btc_value*row['y']
                        btc_value = 0

        if cash == 0:
            portfolio = btc_value*df[end:]['y'][end]
            return (portfolio_value[-1], portfolio_value, transaction_dates, market_portfolio)
        else:
            return (portfolio_value[-1], portfolio_value, transaction_dates, market_portfolio)

    # Run wallet function using user inputs
    st.write(round(wallet(budget, all_score, start=str(d_start), end=str(d_end))[0],2))

    # make function out of datetime/indexing
    # def line_plotting(budget, ):
    #     x = pd.DataFrame(wallet(budget, no_score, start=str(d_start), end=str(d_end))[1],wallet(budget, no_score, start=str(d_start), end=str(d_end))[2])

    x = pd.DataFrame(wallet(budget, no_score, start=str(d_start), end=str(d_end))[1],wallet(budget, no_score, start=str(d_start), end=str(d_end))[2])
    # reset to index, set to to datetime, then set to index
    x = x.reset_index()
    x['index'] = pd.to_datetime(x['index'])
    x = x.set_index('index')
    # st.write(x.index.dtype)
    # st.write(x)

    
    y = pd.DataFrame(wallet(budget, one_score, start=str(d_start), end=str(d_end))[1],wallet(budget, one_score, start=str(d_start), end=str(d_end))[2])
    # reset to index, set to to datetime, then set to index
    y = y.reset_index()
    y['index'] = pd.to_datetime(y['index'])
    y = y.set_index('index')

    z = pd.merge(x,y,left_index=True,right_index=True)
    # fig, ax = plt.subplots(1,1,figsize=(10,8))
    # plt.figure(figsize=(10,8))
    st.line_chart(z)

    # alt.Chart(z).mark_line().encode(
    # x='date',
    # y='price',
    # color='symbol',
    # strokeDash='symbol',)
    # st.area_chart(wallet(budget, one_score, start=str(d_start), end=str(d_end))[2],wallet(budget, one_score, start=str(d_start), end=str(d_end))[1])
    # st.area_chart(wallet(budget, all_score, start=str(d_start), end=str(d_end))[2],wallet(budget, all_score, start=str(d_start), end=str(d_end))[1])
    # plt.show()

    '''
    ### Your wallet gains are...
    '''
    # Run wallet function using user inputs
    # st.write(round(wallet(budget, all_score, start=str(d_start), end=str(d_end))[0],2))
    st.write('...based **only on previous prices**:', round(wallet(budget, no_score, start=str(d_start), end=str(d_end))[0],2), '$')
    st.write('...based on previous price and **one sentiment score**:', round(wallet(budget, one_score, start=str(d_start), end=str(d_end))[0],2), '$')
    st.write('...based on previous price and **four sentiment scores**:', round(wallet(budget, all_score, start=str(d_start), end=str(d_end))[0],2), '$')
    st.write('...if you had simply bought at start-date and sold at end-date (**market performance**):', round(wallet(budget, all_score, start=str(d_start), end=str(d_end))[3],2), '$')

if pages == 'Live Prediction':
    '''
    # 
    # Live predictor indicator v2.0:
    #
    This prediction is based on our wallet 2.0 that you saw in the Main page. We cannot provide an API for our best performing wallet (3.0) due to privacy restrictions from the Augmento team.

    Our API is based on both the 12:00AM UTC **BTC closing value** and the **Fear&Greed index** value that is updated at the same time.
    For this reason our API will be updated every day at 12:05AM.

    This means **optimal use of the API** requires following recommendations at time of release.

    ## Tomorrow's prediction is:
    '''

    # Need to change url once it is changed to docker
    @st.cache
    def response():
        url = 'https://cryptosentiment-cmorf3ig4a-ew.a.run.app'
        return requests.get(url)

    prediction = response().json()
    recommendation = prediction['Recommendation']

    st.write('Predicted BTC Price:', round(prediction['prediction'],2), '$')
    st.write('Closing BTC Price:', round(prediction['current_btc_price'],2), '$')
    st.write('Predicted change:', round(100-((prediction['current_btc_price']/prediction['prediction'])*100),2), '%')
    st.write(f'## Buy/Sell Recommendation: **{recommendation}** !')
    if recommendation == 'Sell':
        st.error('You should sell today, the sooner the better!')
    if recommendation == 'Buy':
        st.success('You should buy bitcoin ASAP!')

    # Timer until next update

    # Graph showing evolution of F&G progress (we have the df so better to plot it ourselves)
    '''
    #
    Here you can see the evolution of the **Fear & Greed Index** since its conception in **2018**:
    '''
    fg_df = pd.read_csv('data/Fear_Greed_df.csv', index_col=0, parse_dates=True)
    st.line_chart(fg_df)


if pages == 'EDA & Metrics':
    '''
    # 
    # Exploratory Data Analysis and Metrics
    #
    **TO-DO**: explain how we determined what messages from Augmenta to take into account and how we defined them (positive, negative, neutral). Should also show distribution of scores per source (data exploration clean has this graph already from yassine)
    
    **Metrics**: MAE and accuracy

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