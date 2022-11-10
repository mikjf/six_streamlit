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

print('Checking downloaded file:')
print(uploaded_file)

# DATE SELECTION
sum_file = 'sum_all_merch'
start_month = '08-2020'