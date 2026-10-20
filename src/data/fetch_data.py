from pathlib import Path
import time

from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashteamstats,
)

SEASON = "2025-26"
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def fetch_player_stats():
    print(f"Fetching player stats for {SEASON}...", flush=True)
    result = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Base",
        per_mode_detailed="PerGame",
        timeout=60,
    )
    df = result.get_data_frames()[0]
    df.to_csv(RAW / "player_stats.csv", index=False)
    print(f"Saved {len(df)} player rows.", flush=True)


def fetch_team_stats():
    print(f"Fetching team stats for {SEASON}...", flush=True)
    result = leaguedashteamstats.LeagueDashTeamStats(
        season=SEASON,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
        timeout=60,
    )
    df = result.get_data_frames()[0]
    df.to_csv(RAW / "team_stats.csv", index=False)
    print(f"Saved {len(df)} team rows.", flush=True)


if __name__ == "__main__":
    fetch_player_stats()
    time.sleep(2)
    fetch_team_stats()
    print("Data download complete.")
