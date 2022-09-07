from typing import List
from pathlib import Path
from dataclasses import dataclass, replace
import pandas as pd


keyword_dicts = {
    "corner": ["hjørnespark"],
    "goal": ["målgivende"],
    "offside": ["offside"],
    "yellow_card": ["gult"],
    "red_card": ["rødt"],
    "goal_shot": ["sjanse", "sjanser"],
    "substitution": ["spillerbytte"],
    "Liverpool": ["liverpool"],
    "ManU": ["manchester", "united"],
}


@dataclass
class Corner:
    time: str


@dataclass
class Goal:
    time: str
    real_time: int = 0#pd.DatetimeIndex = pd.to_datetime("2022-04-19 21:00:00")

@dataclass
class Offside:
    time: str


@dataclass
class YellowCard:
    time: int
    
@dataclass
class RedCard:
    time: str


@dataclass
class GoalShot:
    time: str


@dataclass
class Substitution:
    time: str
    real_time: int = 0  # pd.DatetimeIndex = pd.to_datetime("2022-04-19 21:00:00")


@dataclass
class Match:
    start_time: pd.DatetimeIndex
    half_time_begin: pd.DatetimeIndex
    second_half_begin: pd.DatetimeIndex
    end_time: pd.DatetimeIndex
    goal_events: List[Goal]
    card_events: List[YellowCard]
    sub_events: List[Substitution]


def clean_report(text_report: str) -> List[str]:
    cleaned_reports = []
    list_of_reports = text_report.lower().split()
    for word in list_of_reports:
        cleaned_reports.append(
            ''.join(letter for letter in word if letter.isalnum())
        )
    return cleaned_reports


def clean_reports(game_reports: pd.DataFrame) -> pd.DataFrame:
    cleaned_game_reports = game_reports.copy()
    cleaned_game_reports["reports"] = cleaned_game_reports["reports"].apply(lambda x: clean_report(x))
    return cleaned_game_reports


def keyword_search(search_dict, reports):
    flagged_indices = []
    for index, row in reports.iterrows():
        if set(row["reports"]).intersection(set(search_dict)):
            flagged_indices.append(index)
    return reports.loc[flagged_indices, "game_time"]


def load_in_match_report(path_to_match_report: Path):
    return clean_reports(pd.read_csv(
        path_to_match_report, sep=None, engine="python",
        header=0, encoding='utf-8-sig'))


def initialise_match(start_time: str, reports):
    start_time = pd.to_datetime(start_time)
    half_time_begin = start_time + pd.Timedelta(minutes=46)
    second_half_begin = half_time_begin + pd.Timedelta(minutes=15)
    end_time = second_half_begin + pd.Timedelta(minutes=49)
    goals = []
    cards = []
    subs = []
    for sub in initialise_subs(reports):
        if int(sub.time) > 46:
            subs.append(replace(sub, real_time=(int(sub.time) + 16) * 60))
        else:
            subs.append(replace(sub, real_time=int(sub.time) * 60))
            
    for yellow_card in [87+16, 88+16, 89+16]:
        cards.append(YellowCard(yellow_card*60))
    for goal in initialise_goals(reports):
        if int(goal.time) > 46:
            goals.append(replace(goal, real_time=(int(goal.time)+16)*60))
        else:
            goals.append(replace(goal, real_time=int(goal.time) * 60))
        #goals.append(replace(goal, real_time=start_time+pd.Timedelta(minutes=int(goal.time))))
    return Match(start_time,
                 half_time_begin,
                 second_half_begin,
                 end_time,
                 goals,
                 cards, 
                 subs)


def initialise_subs(reports):
    substitutions = []
    for substitution in keyword_search(keyword_dicts["substitution"], reports):
        substitutions.append(Substitution(substitution))
    return substitutions

def initialise_goals(reports):
    goal_events = []
    for goal_time in keyword_search(keyword_dicts["goal"], reports):
        goal_events.append(Goal(goal_time))
    return goal_events


if __name__ == "__main__":
    path_to_match_reports = Path(__file__).parent.parent / "data" / "game_report.csv"
    reports = load_in_match_report(path_to_match_reports)
    match = initialise_match("2022-04-19 21:00:00", reports)