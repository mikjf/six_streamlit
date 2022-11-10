import pandas as pd
import numpy as np
import scipy
from datetime import datetime
from pmdarima.arima import auto_arima

def run_arima(merchant, merchant_name, seasonality = 12, D_val =1, pred_months = 12):
  # merchant is a df created from row of a data frame we read
  merchant = merchant.set_index('Month')
  merchant['month_index'] = merchant.index.month
  
  # fit the model
  model_final = auto_arima(merchant[[merchant_name]], exogenous=merchant[['month_index']],
                   start_p=1, 
                   max_p=3,
                   start_q=1, 
                   max_q=3,
                   d=None,
                   max_d=3,
                   start_P=1,
                   max_P=3,
                   start_Q=1,
                   max_Q=3,
                   D=D_val,
                   max_D=3,
                   m=seasonality,  
                   seasonal=True,
                   test='adf',
                   trace=False,
                   error_action='ignore',  
                   suppress_warnings=True, 
                   stepwise=True)

  fitted, fitted_CI = model_final.predict_in_sample(return_conf_int=True, exogenous=merchant[['month_index']])

  fitted_df = pd.DataFrame({})
  fitted_df['timestamp'] = merchant.index
  fitted_df['fitted'] = fitted.values
  fitted_df['lower_bound'] = [item[0] for item in fitted_CI]
  fitted_df['upper_bound'] = [item[1] for item in fitted_CI]
  fitted_df = fitted_df.set_index('timestamp')

  # 12 month prediction
  
  # time frame for forecasting
  #!!!!!!!!!!!!!!!!!'2022-10' MANUAL, SHOUOLD BE AUTOMATIC
  dates = pd.Series(pd.period_range('2022-10', freq="M", periods=pred_months))
  #################

  prediction, confint_pred = model_final.predict(n_periods=pred_months, 
                                        return_conf_int=True,
                                        exogenous = dates.dt.month)

  predictions_df = pd.DataFrame({})
  predictions_df['timestamp'] = dates
  predictions_df['predicted'] = prediction.values
  predictions_df['lower_bound'] = [item[0] for item in confint_pred]
  predictions_df['upper_bound'] = [item[1] for item in confint_pred]
  predictions_df = predictions_df.set_index('timestamp')

  return predictions_df, fitted_df


