# Prophet model for time series forecast
from prophet import Prophet

# Data processing
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

def run_prophet(merchant, pred_months = 12):
  # merchant is a df created from row of a data frame we read
  data = merchant.copy(deep=True)
# Change variable names
  my_input = ['ds', 'y'] 
  data.columns = my_input
  data['ds'] = data['ds'].dt.strftime('%Y-%m')
  #print(data)
  # find best hyperparameters and fit the model
  model = Prophet(interval_width=0.99, seasonality_mode='multiplicative')
  
  # Fit the model on the training dataset
  model.fit(data)
  
  # 12 month prediction
  
  # time frame for forecasting
  forecast_df = pd.DataFrame({'ds':pd.date_range(start=data.ds[0], periods =len(data.index)+pred_months, freq="M")})
  results = model.predict(forecast_df)
  results['ds'] = results['ds'].dt.strftime('%Y-%m')
  results = results[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
  columns_pr = ['timestamp', 'predicted','lower_bound', 'upper_bound']
  results.columns = columns_pr
  results = results.set_index('timestamp')
  df_fitted = results.iloc[:-pred_months]
  df_fitted.rename(columns={'predicted':'fitted'}, inplace=True)
  df_predictions = results.iloc[-pred_months:]
  #print(results[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
  #prediction = results[['ds','yhat']].set_index('ds')
  
  # save data
                                        
  #CI_lower = results[['ds','y_lower']].set_index('ds')
  #CI_upper = results[['ds','y_upper']].set_index('ds')
  #CI_width = CI_upper.values - CI_lower.values
  
  #prediction.to_csv(file_path+'data_predictions.csv')

  #CI = pd.concat([CI_lower, CI_upper], axis=1)
  #CI.to_csv(file_path+'CI.csv')
  #CI_upper.to_csv(file_path+'CI_upper.csv')

  #columns_pr = ['predicted', 'lower_bound', 'upper_lower']
  #df_predictions = results.iloc[-pred_months:]
  #df_predictions.columns = columns_pr


  return df_predictions, df_fitted
