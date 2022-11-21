# IMPORTS
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

###################################################################################

# PLOT SIMULATION FUNCTION

def plot_simulation(data, predictions_df_ar, fitted_df_ar, df_fitted_ETS, df_pred_ETS, df_predictions_pr, df_fitted_pr):

    # DATA -> Month to dateTime
    data['Month'] = pd.Series(pd.date_range(str(data.iloc[0]['Month']), freq="M", periods=len(data['Month'])))

    # ARIMA -> Index to DateTime
    predictions_df_ar.index = pd.Series(pd.date_range(str(predictions_df_ar.index[0]), freq="M", periods=len(predictions_df_ar.index)))
    fitted_df_ar.index = pd.Series(pd.date_range(str(fitted_df_ar.index[0]), freq="M", periods=len(fitted_df_ar.index)))

    # ETS -> Index to DateTime
    df_fitted_ETS.index = pd.Series(pd.date_range(str(df_fitted_ETS.index[0]), freq="M", periods=len(df_fitted_ETS.index)))
    df_pred_ETS.index = pd.Series(pd.date_range(str(df_pred_ETS.index[0]), freq="M", periods=len(df_pred_ETS.index)))

    # PR -> Index to DateTime
    df_predictions_pr.index = pd.Series(pd.date_range(str(df_predictions_pr.index[0]), freq="M", periods=len(df_predictions_pr.index)))
    df_fitted_pr.index = pd.Series(pd.date_range(str(df_fitted_pr.index[0]), freq="M", periods=len(df_fitted_pr.index)))

    ###################################################################################

    # SETUP CREATING SUBPLOTS
    fig = make_subplots(rows=3, cols=1, subplot_titles=('ARIMA model', 'ETS model', 'PROPHET model'))

    ###################################################################################

    # ARIMA PLOT

    fig.add_traces(
        [go.Scatter(x=data["Month"], y=data['total'], mode='lines+markers', marker=dict(size=10, color='white'),
                    name='Data', legendgroup = '1',
                    ),
         go.Scatter(x=fitted_df_ar.index, y=fitted_df_ar['fitted'], mode='lines',
                    marker=dict(size=10, color='lightgreen' ), name='Fitted', legendgroup = '1',
                    ),
         go.Scatter(x=predictions_df_ar.index, y=predictions_df_ar.predicted, mode='lines+markers',
                    marker=dict(size=10, color='lightgreen', symbol='triangle-right'), name='Prediction', legendgroup = '1',
                    ),
         go.Scatter(x=predictions_df_ar.index, y=predictions_df_ar['lower_bound'], mode='lines',
              marker=dict(size=1, color='rgba(0,0,0,0)', ), showlegend=False, name='95% CI', legendgroup = '1',
                    ),
         go.Scatter(x=predictions_df_ar.index, y=predictions_df_ar['upper_bound'], mode='lines',
               marker=dict(size=1, color='rgba(0,0,0,0)', ), fill='tonexty', opacity=0.1,
               fillcolor='rgba(204,255,153,0.3)', name='95% CI', legendgroup = '1',
                    hovertemplate='%{y:.0f}'
                    )
         ], rows=1, cols=1,
    )

    ###################################################################################

    # ETS PLOT

    fig.add_traces(
        [go.Scatter(x=data["Month"], y=data['total'], mode='lines+markers', marker=dict(size=10, color='white'),
                    name='Data', legendgroup = '2'),
         go.Scatter(x=df_fitted_ETS.index, y=df_fitted_ETS['fitted'], mode='lines',
                    marker=dict(size=10, color='lightblue' ), name='Fitted', legendgroup = '2'),
         go.Scatter(x=df_pred_ETS.index, y=df_pred_ETS.predicted, mode='lines+markers',
                    marker=dict(size=10, color='lightblue', symbol='triangle-right'), name='Prediction', legendgroup = '2'),
         go.Scatter(x=df_pred_ETS.index, y=df_pred_ETS['lower_bound'], mode='lines',
              marker=dict(size=1, color='rgba(0,0,0,0)', ), showlegend=False, name='95% CI', legendgroup = '2'),
         go.Scatter(x=df_pred_ETS.index, y=df_pred_ETS['upper_bound'], mode='lines',
               marker=dict(size=1, color='rgba(0,0,0,0)', ), fill='tonexty', opacity=0.1,
               fillcolor='rgba(153,255,255,0.3)', name='95% CI', legendgroup = '2')
         ], rows=2, cols=1
    )

    ###################################################################################

    # PROPHET PLOT

    fig.add_traces(
        [go.Scatter(x=data["Month"], y=data['total'], mode='lines+markers', marker=dict(size=10, color='white'),
                    name='Data', legendgroup = '3'),
         go.Scatter(x=df_fitted_pr.index, y=df_fitted_pr['fitted'], mode='lines',
                    marker=dict(size=10, color='yellow', ), name='Fitted', legendgroup = '3'),
         go.Scatter(x=df_predictions_pr.index, y=df_predictions_pr.predicted, mode='lines+markers',
                    marker=dict(size=10, color='yellow', symbol='triangle-right'), name='Prediction', legendgroup = '3'),
         go.Scatter(x=df_predictions_pr.index, y=df_predictions_pr['lower_bound'], mode='lines',
              marker=dict(size=1, color='rgba(0,0,0,0)', ), showlegend=False, name='95% CI', legendgroup = '3'),
         go.Scatter(x=df_predictions_pr.index, y=df_predictions_pr['upper_bound'], mode='lines',
               marker=dict(size=1, color='rgba(0,0,0,0)', ), fill='tonexty', opacity=0.1,
               fillcolor='rgba(255,255,153,0.3)', name='95% CI', legendgroup = '3')
         ], rows=3, cols=1,
    )

    ###################################################################################

    # UPDATING PLOTS LAYOUT

    fig.update_layout(autosize=False, height=1200, plot_bgcolor="#262730", legend_tracegroupgap=320)
    fig.update_xaxes(dtick="M3", tickangle=0, tickformat="%b\n%Y")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(visible=False)

    return fig