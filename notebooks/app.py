# install???
!pip install pmdarima
!pip install prophet

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

# ets
from statsmodels.tsa.api import ExponentialSmoothing
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

# prophet
from prophet import Prophet

# data source and output dir
data_source = '/content/drive/MyDrive/07 - Six - Team only/Processed data/XXX.csv'
output_dir = '/content/drive/MyDrive/Auto-ARIMA90/'

# LAYOUT

# add title and header
st.title("Title")
st.header("Header")

# widget sidebar checkbox to show dataframe
if st.sidebar.checkbox("Show data source"):
    st.header("Header2")
    st.subheader("Subheader:")
    st.dataframe(data=data_source.head())
    #st.table(data=data_source)

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

