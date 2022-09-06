from typing import Dict, List, Tuple
from pathlib import Path
import streamlit as st

from spectators.spectator_class import initialise_spectators, Spectator
from matches.match_report import load_in_match_report, initialise_match

from streamlit_app.pages.data_visualisation_app import data_visualisation
from streamlit_app.pages.data_analysis import data_analysis

path_to_activity = Path(__file__).parent.parent / "data" / "reduced_files"
path_to_demographics = Path(__file__).parent.parent / "data" / "GENEActiv - soccer match - LIV-MANU - 2022-04-19.csv"
path_to_match_reports = Path(__file__).parent.parent / "data" / "game_report.csv"


st.set_page_config(layout="wide")
#configuration of the page

@st.cache
def init_spectators(path_to_activity: Path, path_to_demographics: Path):
        return initialise_spectators(path_to_activity, path_to_demographics)

@st.cache
def init_match(path_to_match_reports: Path):
        reports = load_in_match_report(path_to_match_reports)
        return initialise_match("2022-04-19 21:00:00", reports)

spectators = init_spectators(path_to_activity, path_to_demographics)
match = init_match(path_to_match_reports)


def main_page(spectators, match):
    st.markdown("# Main page")

page_names_to_funcs = {
    "Main Page": main_page,
    "Visualisation": data_visualisation,
    "Data Analysis": data_analysis,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page](spectators, match)
