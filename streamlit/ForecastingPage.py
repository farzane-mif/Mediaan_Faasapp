# Imports
import pandas as pd
import streamlit as st
import io
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
# Local imports
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from core.streamlit.SessionState import SessionState as session_state
import core.streamlit.Util as util


def build_page_forecasting(session: session_state):
    # Load in CSV
    in_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False, type='csv')
    if in_file is not None:
        data = pd.read_csv(io.StringIO(in_file.read().decode('utf-8')), sep=',', index_col=0)
        data.index = pd.to_datetime(data.index)
        graph = util.make_timeseries_graph(data, title="Complete Timeseries Data")
        st.markdown(f"Name: **{in_file.name}**, Datapoints: **{len(data)}**, Date Range: **{data.index[0].date()}** - **{data.index[-1].date()}**")
        st.pyplot(graph)

        # Select Model
        col01, col02, col03 = st.beta_columns((1, 1, 1))
        opt_NONE, opt_ARIMA, opt_PROPHET = ("None", "ARIMA", "Prophet")
        in_model_type = col01.selectbox("Model for Forecasting", (opt_NONE, opt_ARIMA, opt_PROPHET))
        build_model_dict = {opt_ARIMA: build_model_ARIMA,
                            opt_PROPHET: build_model_Prophet}

        # Train/Test threshold
        in_model_train_test_threshold = col02.number_input("Test/Train Threshold", value=50.0, step=1.0, min_value=1.0, max_value=100.0)
        in_model_train_test_threshold = in_model_train_test_threshold / 100
        in_model_horizon = col02.number_input("Horizon", value=50, step=1, min_value=0)
        in_model_train_horizon = col03.number_input("Training Horizon", value=50, step=1, min_value=0)

        training_data = data.iloc[:int(len(data)*in_model_train_test_threshold)]
        prediction_start_date = training_data.index[-1]
        prediction_end_date = data.index[-1]

        # One of the models has been picked and we call the the method to handle it
        if in_model_type != opt_NONE:
            result = build_model_dict[in_model_type](training_data, prediction_start_date, prediction_end_date)
            # Returned result is None if training not done at current point, and [pred, ci] otherwise
            if result is not None:
                prediction, confidence_interval = result
                # Plot
                fig = util_plot(data, prediction, confidence_interval)
                st.pyplot(fig)
                # Evaluate
                # TODO: Error calculations here


# Constructs the UI necessary for training ARIMA and returns result data
def build_model_ARIMA(training_data: pd.DataFrame, prediction_start_date: datetime, prediction_end_date: datetime):
    # Init Return value with default None which is returned assuming the model has not run
    out = None

    # Interface set Parameters
    col01, col02, col03 = st.beta_columns((1, 1, 1))
    in_para_p = col01.number_input("p", value=0, step=1, min_value=0)
    in_para_d = col02.number_input("d", value=0, step=1, min_value=0)
    in_para_q = col03.number_input("q", value=0, step=1, min_value=0)
    col01, col02, col03, col04 = st.beta_columns((1, 1, 1, 1))
    in_para_sp = col01.number_input("P", value=0, step=1, min_value=0)
    in_para_sd = col02.number_input("D", value=0, step=1, min_value=0)
    in_para_sq = col03.number_input("Q", value=0, step=1, min_value=0)
    in_para_ss = col04.number_input("S", value=0, step=1, min_value=0)

    # Train the model
    in_is_train = st.button("Train ARIMA Model")
    if in_is_train:
        model = ARIMA(training_data, order=(in_para_p, in_para_d, in_para_q),
                      seasonal_order=(in_para_sp, in_para_sd, in_para_sq, in_para_ss), enforce_stationarity=False,
                      enforce_invertibility=False)
        model_fit = model.fit()

        pred = model_fit.get_prediction(start=prediction_start_date, end=prediction_end_date, dynamic=False)
        pred_ci = pred.conf_int()

        out = [pred, pred_ci]

    return out


def build_model_Prophet(training_data: pd.DataFrame, prediction_start_date: datetime, prediction_end_date: datetime):
    pass


# Constructs a plot rendering data, prediction and confidence_interval of predicition
def util_plot(data: pd.DataFrame, prediction: pd.DataFrame, confidence_interval=None):
    fig, ax = plt.subplots()
    ax.plot(data, label="Observed")
    prediction.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7)
    if confidence_interval is not None:
        ax.fill_between(confidence_interval.index,
                        confidence_interval.iloc[:, 0],
                        confidence_interval.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    return fig

