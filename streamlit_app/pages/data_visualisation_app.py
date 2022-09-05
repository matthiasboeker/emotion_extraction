from typing import Dict, List, Tuple
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

from spectators.spectator_class import initialise_spectators, Spectator
from matches.match_report import load_in_match_report, initialise_match

def data_visualisation():
    path_to_activity = Path(__file__).parent.parent.parent / "data" / "reduced_files"
    path_to_demographics = Path(__file__).parent.parent.parent / "data" / "GENEActiv - soccer match - LIV-MANU - 2022-04-19.csv"
    path_to_match_reports = Path(__file__).parent.parent.parent / "data" / "game_report.csv"


    def extract_selectable_teams(spectators: List[Spectator]):
        spectators_teams = list(set([spectator.supported_team for spectator in spectators]))
        return sorted([(team, team) if team != "All" else (team, None) for team in spectators_teams + ["All"]])


    def get_spectator(spectator_id: str, spectators_list):
        if spectators:
            return [spectator for spectator in spectators_list if spectator.id == spectator_id][0]
        return None


    def team_filter(spectators: List[Spectator], team_filter: Tuple):
        if team_filter[1]:
            return [spectator.id for spectator in spectators if spectator.supported_team == team_filter[1]]
        return [spectator.id for spectator in spectators]


    def gender_filter(spectators: List[Spectator], gender_filter: Tuple):
        if gender_filter[1]:
            return [spectator.id for spectator in spectators if spectator.gender == gender_filter[1]]
        return [spectator.id for spectator in spectators]


    def filter_spectators(spectators, filter_values: Dict):
        age_filtered = set([spectator.id for spectator in spectators if (spectator.age <= filter_values["age"][1]) and
                           (spectator.age >= filter_values["age"][0])])
        team_filtered = set(team_filter(spectators, filter_values["team"]))
        gender_filtered = set(gender_filter(spectators, filter_values["gender"]))
        filtered_lists = age_filtered & team_filtered & gender_filtered
        if filtered_lists:
            return filtered_lists
        return None


    @st.cache
    def init_spectators(path_to_activity: Path, path_to_demographics: Path):
        return initialise_spectators(path_to_activity, path_to_demographics)


    @st.cache
    def init_match(path_to_match_reports: Path):
        reports = load_in_match_report(path_to_match_reports)
        return initialise_match("2022-04-19 21:00:00", reports)


    st.set_page_config(layout="wide")
    #configuration of the page
    st.title('Emotional Arousment Extraction of a Soccer Game')

    spectators = init_spectators(path_to_activity, path_to_demographics)
    match = init_match(path_to_match_reports)


    with st.sidebar:
        st.header('Select what to display')
        filters = {}
        spectators_ages = [int(spectator.age) for spectator in spectators]
        spectators_teams = extract_selectable_teams(spectators)
        filters["gender"] = st.radio("Select Gender", [ ("All", None), ("Male", "M"), ("Female", "F")],
                                     format_func=lambda x: x[0])
        filters["age"] = st.slider("Select Age", min(spectators_ages),
                               max(spectators_ages), (min(spectators_ages), max(spectators_ages)))
        filters["team"] = st.radio("Select Team", spectators_teams, format_func=lambda x: x[0])

        selected_spectators = filter_spectators(spectators, filters)

        spectators_selected = None
        if selected_spectators:
            spectators_selected = get_spectator(st.selectbox('Spectators', selected_spectators), spectators)

        st.subheader("Visualisation")

        show_ts = st.checkbox("Show time series", )
        show_rolling = st.checkbox("Show rolling mean",)
        show_goals = st.checkbox("Show goals", )

    st.header("Activity Time Series")
    if spectators_selected:
        fig = plt.figure(figsize=(10, 5))
        plt.title(f"Id: {spectators_selected.id}")
        if show_ts:
            plt.plot(spectators_selected.activity)
        if show_rolling:
            plt.plot(spectators_selected.activity.rolling(60).mean(), c = "darkorange")
        if show_goals:
            for goal in match.goal_events:
                plt.axvline(goal.real_time, c="red", linewidth=1)
        st.pyplot(fig)
        st.table(spectators_selected.create_df_for_visualisation())
