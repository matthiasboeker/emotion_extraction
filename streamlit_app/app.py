from typing import Dict, List, Tuple
from pathlib import Path
import streamlit as st
import sys

from spectators.spectator_class import initialise_spectators, Spectator
from matches.match_report import load_in_match_report, initialise_match
from ssa_factorisation.factorise_ssa_matrices import SSA

from streamlit_app.pages.data_visualisation_app import data_visualisation
from streamlit_app.pages.data_analysis import data_analysis

sys.path.append(str(Path(__file__).parent.parent))

path_to_activity = Path(__file__).parent.parent / "data" / "reduced_files"
path_to_demographics = (
    Path(__file__).parent.parent
    / "data"
    / "GENEActiv - soccer match - LIV-MANU - 2022-04-19.csv"
)
path_to_match_reports = Path(__file__).parent.parent / "data" / "game_report.csv"


st.set_page_config(layout="wide")
# configuration of the page


@st.cache
def init_spectators(path_to_activity: Path, path_to_demographics: Path) -> List[Spectator]:
    return initialise_spectators(path_to_activity, path_to_demographics)


@st.cache
def init_ssa_obj(spectators: List[Spectator]) -> Dict[str, SSA]:
    return {spectator.id: SSA.transform_fit(spectator.activity, window_size=60*15, lag=60*5, q=5)
            for spectator in spectators}


@st.cache
def init_match(path_to_match_reports: Path):
    reports = load_in_match_report(path_to_match_reports)
    return initialise_match("2022-04-19 21:00:00", reports)


spectators = init_spectators(path_to_activity, path_to_demographics)
match = init_match(path_to_match_reports)
ssa_objs = init_ssa_obj(spectators)


def main_page(spectators, match, ssa_objs):
    st.markdown("# Main page")


page_names_to_funcs = {
    "Main Page": main_page,
    "Visualisation": data_visualisation,
    "Data Analysis": data_analysis,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page](spectators, match, ssa_objs)
