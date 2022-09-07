from typing import Dict, List, Tuple
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

from streamlit_app.pages.data_visualisation_app import get_spectator
from spectators.spectator_class import Spectator

def normalise_series(series: pd.Series):
    series_max = min(series)
    series_min = max(series)
    return series.apply(lambda x : (x-series_max)/(series_max-series_min))

def data_analysis(spectators, match):
    st.title('Data Analysis')
    ts_df = pd.DataFrame(np.array([spectator.activity for spectator in spectators]
                                  ).T, columns =[spectator.id  for spectator in spectators] )
    correlation_matrix = ts_df.copy()
    correlation_matrix.columns = [spectator.supported_team  for spectator in spectators]
    correlation_matrix = correlation_matrix.corr()
    decomposed_ts = dict(zip([spectator.id  for spectator in spectators], 
                             ts_df.apply(lambda x: seasonal_decompose(x, model='additive', period=60*10),axis=0)))
   
    with st.sidebar:
        st.header('Select what to display')
        show_analysis = st.radio("Show analysis", ("Show correlation matrix", "Show distributions", 
                                                   "Decompose time series", "Autocorrelation"))
    if show_analysis == "Show correlation matrix":
        fig_corr, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(correlation_matrix, ax=ax,annot=True, annot_kws={
                'fontsize': 6,
                'fontweight': 'normal',
                'fontfamily': 'serif'
            })
        st.pyplot(fig_corr)
    if show_analysis == "Show distributions":
        fig_dist, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 12))
        plt.subplots_adjust(hspace=0.5)
        fig_dist.suptitle("Time Series Distributions", fontsize=18)
        
        for ticker, ax in zip(ts_df.columns, axs.ravel()):
            ax.hist(ts_df[str(ticker)].rolling(60).mean()[60:], bins=20, density = True, edgecolor='black',
                    linewidth=0.5, alpha = 0.9,)

            ax.set_title(str(ticker))
        fig_dist.tight_layout()
        st.pyplot(fig_dist)

    if show_analysis == "Decompose time series":
        selected_id = st.sidebar.selectbox("Select Spectator", decomposed_ts.keys())
        fig_decomp, (ax_trend, ax_season, ax_resid) = plt.subplots(3, 1, figsize=(10, 7))
        ax_trend.plot(decomposed_ts[selected_id].trend)
        ax_trend.set_title("Trend")
        ax_season.plot(decomposed_ts[selected_id].seasonal)
        ax_season.set_title("Seasonality")
        ax_resid.scatter(
            np.arange(len(decomposed_ts[selected_id].resid)),
            decomposed_ts[selected_id].resid,
            marker=".")
        ax_resid.set_title("Residuals")
        st.pyplot(fig_decomp)
        spectators_selected = get_spectator(selected_id, spectators)
        st.table(spectators_selected.create_df_for_visualisation())


    if show_analysis == "Autocorrelation":
        selected_id = st.sidebar.selectbox("Select Spectator", decomposed_ts.keys())
        fig_acf, ax = plt.subplots(1,1,figsize=(5,3))
        ax.acorr(ts_df[selected_id], usevlines = True, 
          normed = True, maxlags = 60, 
          lw = 2)
        _, right_x = ax.get_xlim()
        ax.set_xlim(0,right_x)
        st.pyplot(fig_acf)
        spectators_selected = get_spectator(selected_id, spectators)
        st.table(spectators_selected.create_df_for_visualisation())