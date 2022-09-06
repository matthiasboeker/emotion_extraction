from typing import Dict, List, Tuple
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from spectators.spectator_class import Spectator

def data_analysis(spectators, match):
    st.title('Data Analysis')
    ts_df = pd.DataFrame(np.array([spectator.activity for spectator in spectators]).T)
    correlation_matrix = ts_df.copy()
    correlation_matrix.columns = [spectator.supported_team  for spectator in spectators]
    correlation_matrix = correlation_matrix.corr()

    with st.sidebar:
        st.header('Select what to display')
        show_correlation = st.checkbox("Show correlation matrix", )
        show_distributions = st.checkbox("Show distributions", )

    st.header("Analysis")
    if show_correlation and not show_distributions:
        fig_corr, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(correlation_matrix, ax=ax,annot=True, annot_kws={
                'fontsize': 6,
                'fontweight': 'normal',
                'fontfamily': 'serif'
            })
        st.pyplot(fig_corr)
    if show_distributions and not show_correlation:
        fig_dist, g = plt.subplots(figsize=(7,5))
        g = sns.PairGrid(ts_df)
        g.map_diag(plt.hist)
        g.map_offdiag(plt.scatter)
        st.pyplot(fig_dist)
