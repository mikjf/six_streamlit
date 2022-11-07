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

# STREAMLIT
import streamlit as st

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

# PY FILES
from get_sum import get_sum
import auto_arima_single
from auto_arima_single import run_arima_get_rmse
from run_arima_tune_get_pred import run_arima

# PY FILES PROPHET
from auto_single_prophet import run_prophet_get_rmse

###################################################################################

# FILE UPLOADER
uploaded_file = st.file_uploader('Upload a file', type=['csv', 'xlsx'])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, error_bad_lines=True, warn_bad_lines=False)
    except:
        try:
            df = pd.read_excel(uploaded_file)
        except:
            df = pd.DataFrame()
    st.table(df.head())

# DATE SELECTION
sum_file = 'sum_all_merch'
start_month = '08-2020'

###################################################################################

# GET_SUM
try:
    data = get_sum(start_month, df, file_name = sum_file)
    # show get_sum
    #st.table(data)
except:
    pass

# DOWNLOAD FINAL CSV (txt for now and needs to go at the end)
#with open(sum_file+'.csv') as f:
#   st.download_button('Download CSV SUM', f)

# MODELS
models = ['arima', 'ETS', 'prophet']
best_rmse = []

# RUN ARIMA
print(data)
#try:
best_rmse.append(run_arima_get_rmse(merchant = data, merchant_name = 'total', train_test_split = 21, seasonality = 12, D_val =1))
best_rmse.append(run_prophet_get_rmse(merchant = data))
#except:
#    pass

print(best_rmse)

# show best_rmse
if bool(best_rmse):
    st.write(best_rmse)
else:
    pass

print(best_rmse)


#best_rmse.append(run_arima_get_rmse(merchant = data, merchant_name = data.columns.value, train_test_split = 21, seasonality = 12, D_val =1))
#best_rmse.append(run and get rmse from ETS)
#best_rmse.append(run and get rmse from prophet)

# LAYOUT

# widget sidebar checkbox to show dataframe
if st.sidebar.checkbox("Show data source"):
    st.header("Header2")
    st.subheader("Subheader:")
    st.dataframe(data=data_source.head())
    #st.table(data=data_source)

# add title and header
st.title("Title")
st.header("Header")

# setting up columns
#left_column, middle_column, right_column = st.columns([3, 1, 1])

# FIG 1

# first figure title
st.header("Header")

# creating fig 1
fig1 = go.Figure()
fig1.add_traces(
    go.Scatter(
    )
)

