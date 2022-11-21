# IMPORTS
import pandas as pd
import numpy as np
import scipy
from datetime import datetime
from pmdarima.arima import auto_arima
from dateutil.relativedelta import relativedelta

###################################################################################

# RUN_ARIMA FUNCTION
def run_arima(merchant, merchant_name, seasonality = 12, D_val =1, pred_months = 12):

  # merchant is a df with month column and transaction column named merchant_name
  merchant = merchant.set_index('Month')
  
  # fit the model
  model_final = auto_arima(merchant[[merchant_name]],
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

  fitted, fitted_CI = model_final.predict_in_sample(return_conf_int=True)

  fitted_df = pd.DataFrame({})
  fitted_df['timestamp'] = merchant.index
  fitted_df['fitted'] = fitted.values
  fitted_df['lower_bound'] = [item[0] for item in fitted_CI]
  fitted_df['upper_bound'] = [item[1] for item in fitted_CI]
  fitted_df = fitted_df.set_index('timestamp')

  ###################################################################################

  # 12 MONTHS PREDICTION

  start_month_for_pred = str(datetime.strptime(str(merchant.index[-1]), '%Y-%m') + relativedelta(months=+1))[:7]
  dates = pd.Series(pd.period_range(start_month_for_pred, freq="M", periods=pred_months))

  prediction, confint_pred = model_final.predict(n_periods=pred_months, return_conf_int=True)

  predictions_df = pd.DataFrame({})
  predictions_df['timestamp'] = dates
  predictions_df['predicted'] = prediction.values
  predictions_df['lower_bound'] = [item[0] for item in confint_pred]
  predictions_df['upper_bound'] = [item[1] for item in confint_pred]
  predictions_df = predictions_df.set_index('timestamp')

  return predictions_df, fitted_df


