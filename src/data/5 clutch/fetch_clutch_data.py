## fetch_clutch_data

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerclutch

# Minimum clutch minutes required
MIN_CLUTCH_MINUTES = 30

# Fetch 2025-26 regular-season clutch statistics
clutch = leaguedashplayerclutch.LeagueDashPlayerClutch(
    season="2025-26",
    season_type_all_star="Regular Season",
    clutch_time="Last 5 Minutes",
    point_diff=5,
    ahead_behind="Ahead or Behind",
    measure_type_detailed_defense="Base",
    per_mode_detailed="Totals",
    timeout=60,
)

# Convert NBA API response to DataFrame
df = clutch.get_data_frames()[0]

# Keep identifiers and selected statistics
columns = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "GP",
    "MIN",
    "PTS",
    "AST",
]

df = df[columns].copy()

# Apply the minimum clutch-minutes threshold
qualified = df.loc[
    df["MIN"] >= MIN_CLUTCH_MINUTES
].copy()

# Calculate clutch points per 36 minutes
qualified["CLUTCH_PTS_PER_36"] = (
    qualified["PTS"] / qualified["MIN"] * 36
)

# Save one CSV with raw stats and calculated rate
output_file = "player_clutch_rates.csv"
qualified.to_csv(output_file, index=False)

# Summarize results
print(f"Original records: {len(df)}")
print(f"Minimum clutch minutes: {MIN_CLUTCH_MINUTES}")
print(f"Eligible records: {len(qualified)}")
print(f"Excluded records: {len(df) - len(qualified)}")
print(f"Saved: {output_file}")

print("\nTop 10 qualified players by clutch points per 36:")
print(
    qualified.nlargest(10, "CLUTCH_PTS_PER_36")[
        ["PLAYER_NAME", "MIN", "PTS", "CLUTCH_PTS_PER_36"]
    ].to_string(index=False)
)
