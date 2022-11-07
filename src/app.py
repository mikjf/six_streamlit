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
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

# data source and output dir
source_file_path = ''
output_file_path = ''

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
        x=,
        y=,
        mode=,
        name=,
        marker=,
        text=,
        hovertemplate=
    )
)

