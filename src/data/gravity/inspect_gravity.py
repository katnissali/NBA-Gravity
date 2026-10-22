
import pandas as pd
from nba_api.stats.endpoints import gravityleaders

data = gravityleaders.GravityLeaders(
    season="2025-26",
    season_type_all_star="Regular Season",
    league_id="00",
    timeout=60,
)

df = data.leaders.get_data_frame()

print("Players returned:", len(df))
print("\nAvailable columns:")
print(df.columns.tolist())

df["AVGGRAVITYSCORE"] = pd.to_numeric(
    df["AVGGRAVITYSCORE"], errors="coerce"
)

columns = [
    "PLAYERID",
    "FIRSTNAME",
    "LASTNAME",
    "TEAMABBREVIATION",
    "AVGGRAVITYSCORE",
]

print("\nTop 10 gravity scores:")
print(
    df.sort_values(
        "AVGGRAVITYSCORE", ascending=False
    )[columns].head(10).to_string(index=False)
)

print("\nStephen Curry:")
print(
    df.loc[
        df["PLAYERID"].astype(str) == "201939",
        columns
    ].to_string(index=False)
)

print("\nMissing gravity scores:",
      df["AVGGRAVITYSCORE"].isna().sum())


from pathlib import Path

# Preserve the NBA's original columns and add
# convenient names for future dataset matching.
df["PLAYER_ID"] = df["PLAYERID"]
df["GRAV"] = df["AVGGRAVITYSCORE"]

df["SEASON"] = "2025-26"
df["SEASON_TYPE"] = "Regular Season"

output = (
    Path(__file__).resolve().parent
    / "player_gravity_official.csv"
)

df.to_csv(output, index=False)

print(f"\nSaved {len(df)} players to {output.name}")
print("Unique players:", df["PLAYER_ID"].nunique())
print("Missing GRAV:", df["GRAV"].isna().sum())
