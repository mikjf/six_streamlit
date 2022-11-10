# STANDARD IMPORTS
import pandas as pd
import numpy as np
from numpy import array
import scipy
from datetime import datetime
from dateutil.relativedelta import relativedelta
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

# FILE PATHS - YOU NEED TO CHANGE THIS
source = 'data/raw/Time_Series_Merchants_Transactions_Anonymized.csv'
output = 'data/processed/'
start_month = '08-2020'

###################################################################################

# GET_SUM
data = get_sum(source,output,'sum',start_month)
print('Total monthly transaction volumes for', len(data.index), 'months')
print(data)

# get RMSE
print()

models = ['ARIMA', 'ETS', 'PROPHET']
rmse_models = dict.fromkeys(models)

# RUN ARIMA get RMSE
rmse_models['ARIMA'] = (run_arima_get_rmse(merchant=data,merchant_name='total',train_test_split=21,seasonality=12,D_val=1))

# RUN ETS get RMSE
cfg, rmse_models['ETS'] =  run_ETS_get_rmse(data)

# RUN PROPHET get RMSE
rmse_models['PROPHET'] = (run_prophet_get_rmse(merchant=data, train_test_split = 21))

print('List of RMSE per model')
print(rmse_models)

# best = [arima, prophet, ets] depending on the smallest error
best = min(rmse_models, key=rmse_models.get)
print('The best model is {}'.format(best))

# depending on the best model, we get 12 month prediction
# if we want all, we don't need the if/elif
"""if best == 'ARIMA':
    predictions_df, fitted_df = run_arima(data, 'total', seasonality=12, D_val=1, pred_months=12)
    # predictions_df.to_csv(output+str(df.index.max())+'_arima_pred.csv')
    # fitted_df.to_csv(output+str(df.index.max())+'_arima_fit.csv')
elif best == 'ETS':
    df_predictions, df_fitted = predictions(data, cfg)
elif best == 'PROPHET':
    df_predictions = run_prophet(data, pred_months = 12)"""

##testing prophet output
df_predictions_pr, df_fitted_pr = run_prophet(data, pred_months=12)
df_fitted_pr.to_csv(output+str(data.Month.max())+'_Prophet_fit.csv')
df_predictions_pr.to_csv(output+str(data.Month.max())+'_Prophet_pred.csv')

##testing arima output
predictions_df_ar, fitted_df_ar = run_arima(data, 'total', seasonality=12, D_val=1, pred_months=12)
predictions_df_ar.to_csv(output+str(data.Month.max())+'_arima_pred.csv')
fitted_df_ar.to_csv(output+str(data.Month.max())+'_arima_fit.csv')

##testing ETS output
df_predictions_ets, df_fitted_ets = predictions(data, cfg)
df_predictions_ets.to_csv(output+str(data.Month.max())+'_ETS_pred.csv')
df_fitted_ets.to_csv(output+str(data.Month.max())+'_ETS_fit.csv')


#####

