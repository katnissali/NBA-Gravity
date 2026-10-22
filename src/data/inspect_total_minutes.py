
from nba_api.stats.endpoints import leaguedashplayerstats

stats = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2025-26",
    season_type_all_star="Regular Season",
    measure_type_detailed_defense="Base",
    per_mode_detailed="Totals",
    timeout=60,
)

df = stats.get_data_frames()[0]

print("Total records:", len(df))

print("\nExample players:")
print(
    df.loc[
        df["PLAYER_ID"].isin([203932, 201939]),
        ["PLAYER_ID", "PLAYER_NAME", "GP", "MIN"]
    ].to_string(index=False)
)
