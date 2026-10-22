
from nba_api.stats.endpoints import teamplayeronoffsummary

# Golden State Warriors
TEAM_ID = 1610612744

data = teamplayeronoffsummary.TeamPlayerOnOffSummary(
    team_id=TEAM_ID,
    season="2025-26",
    season_type_all_star="Regular Season",
    measure_type_detailed_defense="Advanced",
    per_mode_detailed="Totals",
    timeout=60,
)

on = (
    data.players_on_court_team_player_on_off_summary
    .get_data_frame()
)

off = (
    data.players_off_court_team_player_on_off_summary
    .get_data_frame()
)

columns = [
    "VS_PLAYER_ID",
    "VS_PLAYER_NAME",
    "GP",
    "MIN",
    "OFF_RATING",
]

print("ON records:", len(on))
print("OFF records:", len(off))

print("\nWarriors players ON:")
print(on[columns].head(10).to_string(index=False))

print("\nStephen Curry ON:")
print(
    on.loc[
        on["VS_PLAYER_ID"] == 201939,
        columns
    ].to_string(index=False)
)

print("\nStephen Curry OFF:")
print(
    off.loc[
        off["VS_PLAYER_ID"] == 201939,
        columns
    ].to_string(index=False)
)
