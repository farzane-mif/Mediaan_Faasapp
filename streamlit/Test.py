# Imports
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import io
import random
import datetime
# Local imports
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from core.streamlit.SessionState import SessionState as session_state


# Streamlit Test Page
def build_page_test(session: session_state):
    st.markdown("This is a placeholder page.")


# Streamlit Demo Entry Page
def build_page_entry(session: session_state):
    st.title("Please select one of the features you want demonstrated.")


# Promotion Page
def build_page_promotion(session: session_state):
    st.title("This is the Promotions Page!")
    in_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False, type='csv')
    if in_file is not None:
        st.markdown(in_file.name)
        data = pd.read_csv(io.StringIO(in_file.read().decode('utf-8')), sep=',', index_col=0)
        data.index = pd.to_datetime(data.index)
        zeros = pd.DataFrame(data=range(0, len(data.index)), index=data.index) * 0
        # Marketing Promotion
        st.header("Are you going to start a promotion campaign?")
        _promo_type = st.selectbox("Promotion", ("No", "Yes"))
        if (_promo_type == "Yes"):
            c1, c2 = st.beta_columns([1, 1])
            _p1 = c1.date_input("When your promotion starts?", data.index[0])
            # _p2 = c2.date_input("When your promotion will end?",datetime.date(2020, 1, 1))
            _p2 = c2.number_input("Promotion Duration in Days", value=1)
            # start_index = random.randint(0, len(data.index)-5)
            # end_index = random.randint(start_index+1, len(data.index))
            # number_points = end_index-start_index
            # start_date = data.index[start_index]
            # date_range = pd.date_range(start=start_date, periods=number_points, freq='D')
            # promo = pd.DataFrame(data=range(0, number_points), index=date_range) * 0 + 5
            # data = data + promo
            if (_promo_type == "Yes"):
                promotion_date_range = pd.date_range(start=_p1, freq='D', periods=_p2)
                promotion = pd.DataFrame(data=range(0, _p2), index=promotion_date_range) * 0
                pro1 = promotion + .5
                for i in range(0, _p2):
                    data.loc[promotion_date_range[i]] += 1.5
        st.pyplot(make_timeseries_graph(data))



# Creates and returns graph based on Timeseries data in SessionState
def make_timeseries_graph(timeseries_data: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.set_title('Product')
    ax.plot(timeseries_data)
    return fig

