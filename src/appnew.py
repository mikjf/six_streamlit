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

###################################################################################

# STREAMLIT
import streamlit as st

###################################################################################

#HEADER
st.title('SIX Time Series Prediction')
#st.header('X')
#st.subheader('X')
#st.text('X')
#st.code('for i in range(8): foo()')

###################################################################################

# FILE UPLOADER
uploaded_file = st.file_uploader('Upload dataset to run analysis', type=['csv', 'xlsx'])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, error_bad_lines=True, warn_bad_lines=False)
    except:
        try:
            df = pd.read_excel(uploaded_file)
        except:
            df = pd.DataFrame()
    df = df.set_index(df.columns[0])
    start_month = '08-2020'
    df_dates = pd.Series(pd.period_range(start_month, freq="M", periods=len(df.columns)))
    df.columns = df_dates
    st.text('Data successfully uploaded. ' + str(len(df.axes[0])) + ' merchants x ' + str(len(df.axes[1])) + ' months.\n'
            + 'Period from ' + str(df.columns[0]) + ' to ' + str(df.columns[-1]) + '.')
    #print(df.head())
    #print(df_dates)
    st.text(df.info)
    st.text(df_dates)

###################################################################################

# GET_SUM DISPLAY

if uploaded_file is not None:
    st.text(type(df_dates))
    df_dates = df_dates.astype(str)
    df_dates = pd.to_datetime(df_dates).dt.strftime('%Y-%m')
    data_frame = pd.DataFrame({'Month': df_dates,'total': df.sum(axis=0).values})
    st.table(data_frame)
