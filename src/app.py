# standard imports
import pandas as pd
import numpy as np
from numpy import array
import scipy
from datetime import datetime
import math
import sklearn.metrics
from sklearn.metrics import mean_squared_error

# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# streamlit
import streamlit as st

# warnings
from warnings import catch_warnings
from warnings import filterwarnings

# plotly
import plotly.express as px
import plotly.graph_objects as go

# arima
from pmdarima.arima import auto_arima

# arima py files
from get_sum import get_sum
import auto_arima_single
from auto_arima_single import run_arima_get_rmse
from run_arima_tune_get_pred import run_arima

# ets
from statsmodels.tsa.api import ExponentialSmoothing
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

# prophet
from prophet import Prophet

# file uploader
uploaded_file = st.file_uploader('Upload a file', type=['csv', 'xlsx'])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, error_bad_lines=True, warn_bad_lines=False)
    except:
        try:
            df = pd.read_excel(uploaded_file)
        except:
            df = pd.DataFrame()
    st.table(df)

# ???
sum_file = 'sum_all_merch'
start_month = '08-2020'

# output prediction
try:
    data = get_sum(start_month, df, file_name = sum_file)
    st.table(data)
except:
    pass

# download prediction
with open(sum_file+'.csv') as f:
   st.download_button('Download CSV SUM', f)

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

