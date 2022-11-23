# STANDARD IMPORTS
import pandas as pd
import numpy as np
from numpy import array
import scipy
from datetime import datetime
import math
import sklearn.metrics
from sklearn.metrics import mean_squared_error

# MATPLOTLIB
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# PLOTLY
import plotly.express as px
import plotly.graph_objects as go
from plotly import subplots

# ARIMA
from pmdarima.arima import auto_arima

# ETS
from statsmodels.tsa.api import ExponentialSmoothing
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

# PROPHET
from prophet import Prophet

###################################################################################

# PY FILES ARIMA
from get_sum import get_sum
import auto_arima_single
from auto_arima_single import run_arima_get_rmse
from run_arima_tune_get_pred import run_arima

# PY FILES ETS
from ETS_Function import run_ETS_get_rmse
from ETS_Function import predictions

# PY FILES PROPHET
from auto_single_prophet import run_prophet_get_rmse
from run_prophet_tune_get_pred import run_prophet

# PY FILES PLOT
from Visualization import plot_simulation

###################################################################################

# STREAMLIT
import streamlit as st

###################################################################################

# PAGE SETUP

st.set_page_config(page_title="SIX Time Series Prediction", # page title, displayed on the window/tab bar
                   page_icon="🚀", # favicon: icon that shows on the window/tab bar (tip: you can use emojis)
                   layout="wide", # use full width of the page
                   #menu_items={'About': "Description of the page."}
                   )

###################################################################################

# HEADER AND DISCLAIMER

st.title('Payment Transaction Volumes Forecasting')
st.write("NOTE: The dataset we worked on had already been anonymised since the start of the project. You can run the streamlit app by uploading the [mock dataset](https://github.com/mikjf/six_streamlit/blob/main/mock_data/Mock_Time_Series_Merchants_Transactions_Anonymized.csv) provided. To know everything about the project itself please refer to this [GitHub](https://github.com/mikjf/six_transaction_volumes_forecasting).")

###################################################################################

# START MONTH

start_month = '08-2020'

###################################################################################

# FILE UPLOADER

# file uploader part 1
uploaded_file = st.file_uploader('Upload dataset to run analysis', type=['csv', 'xlsx'])

# progress_bar color
st.markdown(
    """
    <style>
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #ff4b4b , #4bb543);
        }
    </style>""",
    unsafe_allow_html=True,
)

# progress_bar setup
progress_bar = st.progress(0)

# file uploader part 2
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) #, error_bad_lines=True, warn_bad_lines=False)
    except:
        try:
            df = pd.read_excel(uploaded_file)
        except:
            df = pd.DataFrame()

    # progress_bar animation
    for perc_completed in range(100):
        progress_bar.progress(perc_completed+1)

    # file uploader success
    st.success('Dataset uploaded successfully 🎉')

###################################################################################

# DATASET INFO

if uploaded_file is not None:
    df2 = df.copy()
    df2 = df2.set_index(df2.columns[0])
    df2_dates = pd.Series(pd.period_range(start_month, freq="M", periods=len(df2.columns)))
    df2.columns = df2_dates
    c1, c2, c3 = st.columns(3)
    c1.text('🛍️ ' + str(len(df2.axes[0])) + ' merchants')
    c2.text('⌛ ' + str(len(df2.axes[1])) + ' months')
    c3.text('📆 ' + str(df2.columns[0]) + ' to ' + str(df2.columns[-1]))

###################################################################################

# GET_SUM DISPLAY

if uploaded_file is not None:
    data = get_sum(df, 'sum', start_month)
    print(data)
    print()

###################################################################################

# GET_RMSE

if uploaded_file is not None:
    with st.spinner('Running ARIMA, ETS and Prophet models...'):
        models = ['ARIMA', 'ETS', 'PROPHET']
        rmse_models = dict.fromkeys(models)

        # RUN ARIMA get RMSE
        rmse_models['ARIMA'] = (run_arima_get_rmse(merchant=data, merchant_name='total', train_test_split=21, seasonality=12, D_val=1))

        # RUN ETS get RMSE
        cfg, rmse_models['ETS'] = run_ETS_get_rmse(data)

        # RUN PROPHET get RMSE
        rmse_models['PROPHET'] = (run_prophet_get_rmse(merchant=data, train_test_split=21))

        # python console print check
        print('List of RMSE per model')
        print(rmse_models)
        print()

        # best model depending on the smallest error
        best = min(rmse_models, key=rmse_models.get)

        # python console print check
        print('The best model is {}'.format(best))
        print()

        st.success("Which model has the lowest RMSE? Let's find out! 🥇")

###################################################################################

# ALL MODELS 12M PREDICTIONS

if uploaded_file is not None:

    # ARIMA output
    predictions_df_ar, fitted_df_ar = run_arima(data, 'total', seasonality=12, D_val=1, pred_months=12)

    # ETS output
    df_predictions_ets, df_fitted_ets = predictions(data, cfg)

    # PROPHET output
    df_predictions_pr, df_fitted_pr = run_prophet(data, pred_months=12)

    # python console print check
    print('All models 12 months prediction completed')
    print()

###################################################################################

# PLOTTING

    # creating matplotlib plots
    fig = plot_simulation(data, predictions_df_ar, fitted_df_ar, df_fitted_ets, df_predictions_ets,
                    df_predictions_pr, df_fitted_pr)

    # winner sentence
    st.text('The RMSE for ' + list(rmse_models.keys())[0] + ' is ' + str(round(list(rmse_models.values())[0])))
    st.text('The RMSE for ' + list(rmse_models.keys())[1] + ' is ' + str(round(list(rmse_models.values())[1])))
    st.text('The RMSE for ' + list(rmse_models.keys())[2] + ' is ' + str(round(list(rmse_models.values())[2])))
    st.subheader('And the model with the lower RMSE is... {} 🏆'.format(best))
    st.subheader('Prediction details by model ⬇️')

    # importing plots to streamlit
    st.plotly_chart(fig, use_container_width=True)

    # python console print check
    print('All models plotting completed')
    print()

###################################################################################

# RUN COMPLETED

    st.balloons()