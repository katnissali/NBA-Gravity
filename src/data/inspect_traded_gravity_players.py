"""Read-only inspection of ON/OFF factors for players who played for multiple NBA teams.

Place in NBA-Gravity/src/data/ and run:
    python src/data/inspect_traded_gravity_players.py
No files are written or changed.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # Assumes this script lives in src/data/


def read_csv(filename):
    matches = sorted(ROOT.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {filename} under {ROOT}; found: {matches}"
        )
    return pd.read_csv(matches[0])


off = read_csv("player_off_rating_on_off.csv")
efg = read_csv("player_teammate_efg_on_off.csv")
grav = read_csv("player_gravity_official.csv")

for frame in (off, efg):
    frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="raise").astype(int)

grav_ids = set(pd.to_numeric(grav["PLAYERID"], errors="raise").astype(int))
team_counts = off.groupby("PLAYER_ID")["TEAM_ID"].nunique()
traded = sorted(pid for pid in team_counts[team_counts > 1].index if pid in grav_ids)

print(f"GRAV players with multiple teams: {len(traded)}/{len(grav_ids)}")

examples = [
    (201935, "James Harden"),
    (1628379, "Luke Kennard"),
    (1626145, "Tyus Jones"),
]

for player_id, name in examples:
    rows = off.loc[off["PLAYER_ID"] == player_id, [
        "PLAYER_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GP_ON",
        "MIN_ON", "OFF_RATING_DIFF",
    ]].merge(
        efg[["PLAYER_ID", "TEAM_ID", "EFG_DIFF_PP"]],
        on=["PLAYER_ID", "TEAM_ID"],
        how="left",
        validate="one_to_one",
    )
    print(f"\n{name} (NBA player ID {player_id})")
    if rows.empty:
        print("  Player not found in OFF_RATING file.")
    else:
        print(rows[["TEAM_ABBREVIATION", "GP_ON", "MIN_ON", "OFF_RATING_DIFF", "EFG_DIFF_PP"]].to_string(index=False))

print("\nInspection only: no team stints were combined or files changed.")
