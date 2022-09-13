from typing import Dict, List, Tuple
from pathlib import Path
import streamlit as st
import sys
import pickle
sys.path.append(str(Path(__file__).parent.parent))

from spectators.spectator_class import initialise_spectators, Spectator
from matches.match_report import load_in_match_report, initialise_match
from ssa_factorisation.factorise_ssa_matrices import SSA

from streamlit_app.pages.data_visualisation_app import data_visualisation
from streamlit_app.pages.data_analysis import data_analysis


path_to_pickles = Path(__file__).parent.parent / "data"

st.set_page_config(layout="wide")
# configuration of the page


@st.cache
def load_in_pickles(path_to_pickles: Path, file_names: List[str]):
    spectators = pickle.load(open(path_to_pickles/ file_names[0], "rb"))
    match = pickle.load(open(path_to_pickles/ file_names[1], "rb"))
    ssa_objs = pickle.load(open(path_to_pickles/ file_names[2], "rb"))
    return spectators, match, ssa_objs


spectators, match, ssa_objs = load_in_pickles(path_to_pickles, ["spectators.pkl", "match.pkl", "ssa_objs.pkl"])


def main_page(spectators, match, ssa_objs):
    st.markdown("# Main page")


page_names_to_funcs = {
    "Main Page": main_page,
    "Visualisation": data_visualisation,
    "Data Analysis": data_analysis,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page](spectators, match, ssa_objs)
