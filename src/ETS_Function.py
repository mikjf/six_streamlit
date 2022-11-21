# IMPORTS
from statsmodels.tsa.api import ExponentialSmoothing
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

###################################################################################

# EXP_SMOOTHING_FORECAST FUNCTION

def exp_smoothing_forecast(history, config):
    t, d, s, p, b, r, sm = config
    history = array(history)
    model = ExponentialSmoothing(history, trend=t, damped_trend=d, seasonal=s, seasonal_periods=p)
    model_fit = model.fit(smoothing_level=sm, optimized=True, remove_bias=r) #, use_boxcox=b
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]

###################################################################################

# TRAIN_TEST_SPLIT FUNCTION

def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

###################################################################################

# MEASURE_RMSE FUNCTION

def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

###################################################################################

# MEASURE_RMSE FUNCTION

def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    train, test = train_test_split(data, n_test)
    history = [x for x in train]
    for i in range(len(test)):
        yhat = exp_smoothing_forecast(history, cfg)
        predictions.append(yhat)
        history.append(test[i])
    error = measure_rmse(test, predictions)
    return error

###################################################################################

# SCORE_MODEL FUNCTION

def score_model(data, n_test, cfg, debug=False): #
    result = None
    key = str(cfg)
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        try:
            result = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
    return (key, result)

###################################################################################

# GRID_SEARCH FUNCTION (looking for the model with the lowest error)

def grid_search(data, cfg_list, n_test, parallel=False):
    scores = None
    if parallel:
        # executes configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # removes empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores

###################################################################################

# CONFIGURATIONS TO TRAIN

###################################################################################

# EXP_SMOOTHING_CONFIGS FUNCTION

def exp_smoothing_configs(seasonal=[0 ,4, 6, 8, 12]):
    models = list()
    # define config lists
    t_params = ['add', 'mul', None]
    d_params = [True, False]
    s_params = ['add', 'mul', None]
    p_params = seasonal
    b_params = [True, False]
    r_params = [True, False]
    sm_params = [0.1, 0.25, 0.5, 0.75, 1]
    # create config instances
    for t in t_params:
        for d in d_params:
            for s in s_params:
                for p in p_params:
                    for b in b_params:
                        for r in r_params:
                            for sm in sm_params:
                                cfg = [t, d, s, p, b, r, sm]
                                models.append(cfg)
    return models

###################################################################################

# RUN_ETS_GET_RMSE FUNCTION (extracting lowest error -> returns RMSE)

def run_ETS_get_rmse(df, n_test=5):
    data = df.copy(deep=True)
    data = data.set_index('Month')
    data = data.squeeze()
    list_s = "][' "
    cfg_list = exp_smoothing_configs()
    scores = grid_search(data, cfg_list, n_test)
    for cfg, error in scores[:1]:
        for i in list_s:
            cfg = cfg.replace(i, '')
        cfg1= cfg.split(',')
    for i, n in enumerate(cfg1):
        if n == 'False':
            cfg1[i] = False
        elif n == 'True':
            cfg1[i] = True
        try:
            float(n)
            cfg1[i] = float(n)
            result = True
        except:
            continue
    return cfg1, error

###################################################################################

# MODELING_ETS FUNCTION (modeling and fitting)

def modeling_ETS(data, cfg):
    data = data.copy(deep=True)
    data = data.set_index('Month')
    data = data.squeeze()
    t, d, s, s_p, bx, b, sm = cfg
    model = ExponentialSmoothing(data, trend=t, damped_trend=d, seasonal=s, seasonal_periods=s_p)
    fit = model.fit(smoothing_level=sm, remove_bias=b) # use_boxcox=bx
    return model, fit

###################################################################################

# BOUNDARIES FUNCTION (drawing the boundaries for plotting)

def boundaries(simulations):
    simulations_tr = simulations.transpose(copy=True)
    simulations_tr_summary = simulations_tr.describe(percentiles=[.025, .5, .975])
    simulations_summary = simulations_tr_summary.transpose(copy=True)
    y1 = simulations_summary['2.5%']
    y2 = simulations_summary['97.5%']
    CI_ETS = pd.concat([y1, y2], axis=1)
    return CI_ETS

###################################################################################

# PREDICTIONS FUNCTION

def predictions(data, cfg, num_sim=12):
  _, fit = modeling_ETS(data, cfg)
  fitted = pd.DataFrame(fit.fittedvalues, columns=['Transactions'])
  CI_fitted = fit.simulate(nsimulations=data.shape[0],
                            repetitions=1000,
                            anchor='start'
                            )
  columns_1=['fitted', 'lower_bound', 'upper_bound']
  columns_2=['predicted', 'lower_bound', 'upper_bound']
  upper_ci = CI_fitted.quantile(q=0.975, axis='columns')
  lower_ci = CI_fitted.quantile(q=0.025, axis='columns')
  CI_fitted = pd.concat([lower_ci, upper_ci], axis=1)
  dp = [fitted, CI_fitted]
  df_fitted = pd.concat(dp, join='inner', axis=1)
  df_fitted.columns = columns_1
  df_fitted = df_fitted.rename_axis('timestamp')

  simulations = fit.simulate(num_sim, repetitions=1000, error="mul")
  predictions = fit.forecast(num_sim)
  CI_predictions = boundaries(simulations)
  ds = [predictions, CI_predictions]
  df_predictions = pd.concat(ds, join='inner', axis=1)
  df_predictions.columns = columns_2
  df_predictions = df_predictions.rename_axis('timestamp')

  return df_predictions, df_fitted

###################################################################################

# PLOT_SIMULATION FUNCTION

def plot_simulation(data, cfg):
    t = input('Select the name of the graph: ')
    p = input('Select the name of the forecasting label: ')
    model, fit = modeling_ETS(data, cfg)
    simu = simulations(fit)
    y1, y2, x = boundaries(simu)
    ax = data.plot(
        figsize=(10, 6),
        marker="o",
        color="black",
        title=t,
        label='Sum of all Merchants'
    )
    ax.set_ylabel("Nº of Transactions ")
    ax.set_xlabel("Year")  # Year
    fit1.fittedvalues.plot(ax=ax, style="--", color="green")
    ax.fill_between(x, y1, y2, alpha=0.2, color='g', animated=True)
    fit1.forecast(12).rename(p).plot(
        ax=ax, style="--", marker=">", color="green", legend=True
    )