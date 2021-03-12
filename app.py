import streamlit as st
import time
import datetime
import pandas as pd
import numpy as np
from PIL import Image
import requests
import base64

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
    
    Special shout-out to Clementine Contat for guidance through the project!
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

    1. [Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/) : This index takes into account not only social media sentiment (accounts for 15% of overall value) but also other economic indicators like **Volatility** (25%) or **Market Momentum/Volume** (25%). This gives an added economical insight to our predictions.
    2. [Augmento Sentiment Data](https://www.augmento.ai/) : This source provides labelled messages from different sources (twitter, reddit and bitcoin talk) which we then filtered for **positive, negative** and **neutral emotions**. More on our methodology can be found in EDA and Metrics page.

    We used Facebook's Prophet model due to its predictive capacity, seasonality fits, and ease of use.
    Below you will find contrasting results for predictions with and without the use of sentiment indicators.

    '''

    # Interactive investment wallets:

    '''
    # 
    # Interactive investment wallets
    '''
    # Load csv with predictions
    no_score = pd.read_csv('data/predictions_no_score.csv')
    one_score = pd.read_csv('data/predictions_fear_greed_score.csv')
    all_score = pd.read_csv('data/predictions_all_scores_best.csv')

    '''
    ### These wallets will allow **you** to explore the effects of the analyzed sentiment indexes on your *potential* investments over time... If they were done in the past.
    ###
    '''
    # creating expander object
    expander_creators = st.beta_expander("Wallet information breakdown (click for more)", expanded=False)
    with expander_creators:
        # content within the expander
        # a1, a2 = st.beta_columns((1,1))
        # a1.image('https://res.cloudinary.com/dbxctsqiw/image/upload/v1615466629/SmArt/Jae_s0lcs8.png')
        # a2.image('https://res.cloudinary.com/dbxctsqiw/image/upload/v1615466627/SmArt/ed_ljtaqb.png')
        '''
        The results below show 3 example wallet performances *plus* simple market performance. The first wallet (**no signal**) only accounts for previous price fluctuations, which our time-series Prophet model took into account for the prediction.
        The second wallet (**one signal**) accounts for the Fear&Greed index as an exogenous variable. The third (**four signals**), and best performing wallet, also takes into account Augmento data for which we collected social media sentiment scores from twitter, reddit, and bitcoin talk messages. 
        By taking into account Fear&Greed and Augmento data as **exogenous variables** our final wallet becomes **the best performing** one.
        '''
    '''
    ###
    Please choose a **time frame** and wallet **budget** (USD $):
    '''

    # Function for wallet/portfolio investing
    def wallet(budget, df, start='2019-06-19', end='2021-01-31'):
        start = df[df['ds']==start].index[0]
        end = df[df['ds']==end].index[0]
        new_df = df[start:end]
        cash = budget
        btc_value = 0
        market_portfolio = (budget/new_df['y'][start])*new_df['y'][:end]
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

    # user inputs
    budget = st.number_input('Insert initial investment (USD $)', value = 100, min_value = 100, max_value = 10_000_000)
    d_start = st.date_input('Start of investment', datetime.datetime(2019, 6, 20), min_value=datetime.datetime(2019, 6, 20), max_value=datetime.datetime(2021, 1, 31))
    d_end = st.date_input('End of investment', datetime.datetime(2021, 1, 31), min_value=datetime.datetime(2019, 6, 20), max_value=datetime.datetime(2021, 1, 31))

    # Run wallet function using user inputs for plotting
    def prediction_plot_df(prediction_df, market=False):
        if market == False:
            x = pd.DataFrame(wallet(budget, prediction_df, start=str(d_start), end=str(d_end))[1],wallet(budget, prediction_df, start=str(d_start), end=str(d_end))[2])
            x = x.reset_index()
            x['index'] = pd.to_datetime(x['index'])
            x = x.set_index('index')
        else:
            x = pd.DataFrame(list(wallet(budget, prediction_df, start=str(d_start), end=str(d_end))[3]),wallet(budget, prediction_df, start=str(d_start), end=str(d_end))[2])
            x = x.reset_index()
            x['index'] = pd.to_datetime(x['index'])
            x = x.set_index('index')
        return x
 
    graph_df = pd.merge(prediction_plot_df(no_score),prediction_plot_df(one_score),left_index=True,right_index=True)
    graph_df = pd.merge(graph_df,prediction_plot_df(all_score),left_index=True,right_index=True)
    graph_df = pd.merge(graph_df,prediction_plot_df(all_score, market=True),left_index=True,right_index=True)
    graph_df.columns = ['No Signal', 'One Signal', 'Four Signals','Market']
    st.line_chart(graph_df)

    # Potential new graph style
    # alt.Chart(z).mark_line().encode(
    # x='date',
    # y='price',
    # color='symbol',
    # strokeDash='symbol')

    '''
    ### Your wallet gains are...
    '''
    # Run wallet function using user inputs
    # st.write(round(wallet(budget, all_score, start=str(d_start), end=str(d_end))[0],2))
    st.write('...based **only on previous prices** (no signal):', round(wallet(budget, no_score, start=str(d_start), end=str(d_end))[0],2), '$')
    st.write('...based on previous price and **one sentiment score** (one signal):', round(wallet(budget, one_score, start=str(d_start), end=str(d_end))[0],2), '$')
    st.write('...based on previous price and **four sentiment scores** (four signals):', round(wallet(budget, all_score, start=str(d_start), end=str(d_end))[0],2), '$')
    st.write('...if you had simply bought at start-date and sold at end-date (**market performance**):', round(wallet(budget, all_score, start=str(d_start), end=str(d_end))[3][len(wallet(budget, all_score, start=str(d_start), end=str(d_end))[3])],2), '$')

if pages == 'Live Prediction':
    '''
    # 
    # Live predictor indicator v2.0:
    #
    This prediction is based on our **one signal** wallet that you saw in the Main page. We cannot provide an API for our best performing wallet (3.0) due to privacy restrictions from the Augmento team.

    Our API is based on both the **BTC closing value** and the **Fear&Greed index** value, which are both updated at the **12:00AM UTC**.
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
    ##
    This section is intended to give *insight* into our aquisition of data, our feature engineering methodologies, and the optimum performance of our model. We hope this helps users and their conviction to trust our model predictions.
    ## Sentiment indexes:
    ### Augmento Social Media Scores
    Augmento is an API that constantly collects cryptocurrency related conversations from Twitter, Reddit & Bitcointalk. Using a **classifier trained on crypto specific language**, the data is analyzed according to **93 sentiments and topics**. 
    From there we selected specific topics that we believe can be translated as *“price driving topics”* or *“price lowering topic”* for bitcoins. This method allowed to create years of stable data suitable for backtesting.
    
    Below you will find our methodolgy for **filtering Augmento's social media messages**:
    '''
    # Social Media Postive/Negative filtering and category count
    augmento_count_image = Image.open('images/augmento-count-categories.png')
    st.image(augmento_count_image,caption='Augmento Categories Count', use_column_width=True)

    '''
    Below is the final **score progression over time** of each social media platform after our **filtering for positive and negative posts**. As you can see each platform shows a different general tendency towards Bitcoin, but **never static**.
    '''
    # Distribution of augmento social media scores AFTER filtering
    augmento_image = Image.open('images/augmento-final-scores-graph.png')
    st.image(augmento_image,caption='Augmento Social Media Scores after filtering', use_column_width=True)

    '''
    ### Index Corrolation
    Below you will find a heatmap showing **corrolations** between our exogenous variables (the Fear&Greed index and the three social media scores from Augmento) and the value of Bitcoin.
    '''
    # Heatmap for corrolations
    heatmap_image = Image.open('images/heatmap2.png')
    st.image(heatmap_image,caption='Heatmap BTC prices and indexes', use_column_width=True)

    '''
    ## Prophet model:
    Prophet uses a decomposable time series model with three main model components: trend, seasonality, and holidays. They are combined in the following equation:
    '''
    # For mathematical functions use latex
    st.latex(r'''
        y(t)= g(t) + s(t) + h(t) + εt
        ''')
    '''
    > * **g(t)**: piecewise linear or logistic growth curve for modeling non-periodic changes in time series
    > * **s(t)**: periodic changes (e.g. weekly/yearly seasonality)
    > * **h(t)**: effects of holidays (user provided) with irregular schedules
    > * **εt**: error term accounts for any unusual changes not accommodated by the model
    
    ### Seasonality:
    Thanks to prophet built-in methods we were able to extract a better performance from our model. Part of prophet's strenghts is its capacity to capture seasonality, so by using:''' 
    st.code('model.plot_components(forecast)')
    '''
    we were able to extract that the BTC value data had an inherent **weekly seasonality**:
    '''
    seasonality_image = Image.open('images/weekly-seasonality.png')
    st.image(seasonality_image,caption='Weekly Value Forecast', use_column_width=True)
    '''
    This shows that buyers should actually be targetting Thursday's as the best day to buy!

    ### Model Prediction Accuracy and Mean Absolute Error (MAE):
    '''
    # Load csv with predictions
    no_score = pd.read_csv('data/predictions_no_score.csv')
    one_score = pd.read_csv('data/predictions_fear_greed_score.csv')
    all_score = pd.read_csv('data/predictions_all_scores_best.csv')

    def metrics(pred_df):
        accuracy = pred_df['correct_pred'].sum()/len(pred_df['correct_pred'])
        MAE = pred_df['mae'].mean()
        return accuracy, MAE
    
    st.write('No Signal Accuracy:', round(metrics(no_score)[0],3))
    st.write('No Signal MAE:', round(metrics(no_score)[1],3))
    st.write('One Signal Accuracy:', round(metrics(one_score)[0],3))
    st.write('One Signal MAE:', round(metrics(one_score)[1],3))
    st.write('Four Signal Accuracy:', round(metrics(all_score)[0],3))
    st.write('Four Signal MAE:', round(metrics(all_score)[1],3))

    '''

    '''
    '''
    # 
    # Future Steps
    ### - Getting augmento live api
    ### - Do our own sentiment analysis via web scraping to avoid needing indexes
    ### - Trying out predictions with other cryptocurrencies 
    ### - Converting the model into hourly predictions for better accuracy
    '''