from typing import List
import os
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from matches.match_report import load_in_match_report

@dataclass(frozen=True)
class Spectator:
    id: str
    gender: str
    age: int
    supported_team: str
    length_of_team_support: int
    team_success_importance: int
    team_support: int
    opponent_dislike: int
    activity: pd.Series
    time: pd.Series

    def create_df_for_visualisation(self):
        return pd.DataFrame(
            {
                "Gender": self.gender,
                "Age": self.age,
                "Supported team": self.supported_team,
                "Supports team since": self.length_of_team_support,
                "Team success importance": self.team_success_importance,
                "Dislike of opponent team": self.opponent_dislike,
            },
            index=pd.Index([self.id]),
        )


def extract_id(file_name: str) -> str:
    first_dashes = 3
    return file_name[first_dashes:].split("_")[0]


def initialise_spectator(
    spectator_id: str, path_to_spectator_activity: Path, demographics_df: pd.DataFrame
) -> Spectator:
    activity = pd.read_csv(
        path_to_spectator_activity, sep=None, engine="python", dtype={"time": str}
    )
    spectator_demographics = demographics_df.loc[
        demographics_df["﻿GENEActiv ID"] == spectator_id, :
    ]
    return Spectator(
        spectator_id,
        spectator_demographics["Gender"].iat[0],
        spectator_demographics["Age"].iat[0],
        spectator_demographics["My Team"].iat[0],
        spectator_demographics["Since when have you been a fan of YOUR team"].iat[0],
        spectator_demographics["How important to YOU is it that YOUR team wins?"].iat[
            0
        ],
        spectator_demographics[
            "How strongly do YOU see yourself as a fan of YOUR team?"
        ].iat[0],
        spectator_demographics["How much do YOU dislike the other team?"].iat[0],
        activity["signal"],
        activity["time"],
    )


def initialise_spectators(
    path_to_activity: Path, path_to_demographics: Path
) -> List[Spectator]:
    demographics = pd.read_csv(
        path_to_demographics, sep=None, engine="python", dtype={"﻿GENEActiv ID": str}
    )
    spectators = []
    for file in [
        file for file in os.listdir(path_to_activity) if file.find(".csv") != -1
    ]:
        spectators.append(
            initialise_spectator(
                extract_id(file), path_to_activity / file, demographics
            )
        )
    return spectators