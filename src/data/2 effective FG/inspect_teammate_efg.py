
from nba_api.stats.endpoints import (
    teamplayeronoffdetails,
    teamplayerdashboard,
)

SEASON = "2025-26"
TEAM_ID = 1610612744  # Golden State Warriors
PLAYER_ID = 201939    # Stephen Curry

settings = {
    "team_id": TEAM_ID,
    "season": SEASON,
    "season_type_all_star": "Regular Season",
    "measure_type_detailed_defense": "Base",
    "per_mode_detailed": "Totals",
    "timeout": 60,
}

# 1. Get team shooting totals with each player ON/OFF
details = teamplayeronoffdetails.TeamPlayerOnOffDetails(
    **settings
)

on = details.players_on_court_team_player_on_off_details.get_data_frame()
off = details.players_off_court_team_player_on_off_details.get_data_frame()

# 2. Get individual player shooting totals for this team
dashboard = teamplayerdashboard.TeamPlayerDashboard(
    **settings
)

players = dashboard.players_season_totals.get_data_frame()

# 3. Locate Stephen Curry in each dataset
def get_player_row(df, id_column):
    matches = df.loc[df[id_column] == PLAYER_ID]

    if len(matches) != 1:
        raise ValueError(
            f"Expected 1 Curry row, found {len(matches)}"
        )

    return matches.iloc[0]


curry_on = get_player_row(on, "VS_PLAYER_ID")
curry_off = get_player_row(off, "VS_PLAYER_ID")
curry_stats = get_player_row(players, "PLAYER_ID")

# 4. Calculate effective field goal percentage
def efg(fgm, fga, fg3m):
    if fga <= 0:
        raise ValueError("Field-goal attempts must be positive")
    return (fgm + 0.5 * fg3m) / fga


# Remove Curry's personal shots from Warriors ON totals
teammate_fgm = curry_on["FGM"] - curry_stats["FGM"]
teammate_fga = curry_on["FGA"] - curry_stats["FGA"]
teammate_fg3m = curry_on["FG3M"] - curry_stats["FG3M"]

if min(teammate_fgm, teammate_fga, teammate_fg3m) < 0:
    raise ValueError("Player totals exceed team ON totals")

efg_on = efg(
    teammate_fgm,
    teammate_fga,
    teammate_fg3m,
)

# Curry isn't shooting while OFF, so no subtraction
efg_off = efg(
    curry_off["FGM"],
    curry_off["FGA"],
    curry_off["FG3M"],
)

difference = efg_on - efg_off

print("\nWarriors ON shooting totals:")
print(curry_on[["FGM", "FGA", "FG3M"]].to_string())

print("\nCurry individual shooting totals:")
print(curry_stats[["FGM", "FGA", "FG3M"]].to_string())

print("\nWarriors OFF shooting totals:")
print(curry_off[["FGM", "FGA", "FG3M"]].to_string())

print("\n--- TEAMMATES-ONLY eFG% ---")
print(f"Curry ON:  {efg_on:.4%}")
print(f"Curry OFF: {efg_off:.4%}")
print(f"Difference: {difference * 100:+.2f} percentage points")
