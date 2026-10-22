
import pandas as pd
from nba_api.stats.endpoints import (
    leaguegamefinder,
    playbyplayv3,
)

# Retrieve a game from the 2025-26 regular season
finder = leaguegamefinder.LeagueGameFinder(
    season_nullable="2025-26",
    season_type_nullable="Regular Season",
    league_id_nullable="00",
    timeout=60,
)

games = finder.get_data_frames()[0]
game_id = str(games.iloc[0]["GAME_ID"])

print("Inspecting game:", game_id)

# Retrieve the game's play-by-play events
pbp = playbyplayv3.PlayByPlayV3(
    game_id=game_id,
    timeout=60,
)

df = pbp.get_data_frames()[0]

# Ensure events are in chronological order
df = df.sort_values("actionId").reset_index(drop=True)

print("\nEvent types:")
print(df["actionType"].value_counts())

# Show events surrounding the first 5 timeouts
timeout_indices = df.index[
    df["actionType"] == "Timeout"
]

columns = [
    "period",
    "clock",
    "actionType",
    "teamTricode",
    "playerName",
    "description",
]

print("\nEvents preceding timeouts:")

for idx in timeout_indices[:5]:
    print("\n--- Timeout sequence ---")
    start = max(0, idx - 2)
    print(
        df.loc[start:idx, columns]
        .to_string(index=False)
    )


# Inspect team identifiers for timeout events
print("\nTimeout team identification:")

timeout_columns = [
    "actionId",
    "teamId",
    "teamTricode",
    "description",
]

print(
    df.loc[
        df["actionType"] == "Timeout",
        timeout_columns
    ].to_string(index=False)
)


# Identify the two teams in our test game
timeout_teams = {
    "clippers": "LAC",
    "warriors": "GSW",
}

timeout_responses = []

# Examine each timeout in chronological event order
for idx in df.index[df["actionType"] == "Timeout"]:

    if idx == 0:
        continue

    previous = df.iloc[idx - 1]
    timeout = df.iloc[idx]

    # Timeout must immediately follow a made basket
    if previous["actionType"] != "Made Shot":
        continue

    description = str(timeout["description"]).lower()

    # Identify the team calling the timeout
    timeout_team = None

    for name, abbreviation in timeout_teams.items():
        if description.startswith(name + " timeout"):
            timeout_team = abbreviation
            break

    scoring_team = previous["teamTricode"]

    # Count only opposing-team timeouts
    if timeout_team and timeout_team != scoring_team:
        timeout_responses.append({
            "PLAYER_ID": previous["personId"],
            "PLAYER_NAME": previous["playerName"],
            "SCORING_TEAM": scoring_team,
            "TIMEOUT_TEAM": timeout_team,
        })

# Display qualifying events
results = pd.DataFrame(timeout_responses)

print("\nQualifying opposing timeout responses:")
print(results.to_string(index=False))

print("\nTotal qualifying responses:", len(results))