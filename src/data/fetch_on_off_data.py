from pathlib import Path
import time
import pandas as pd
from nba_api.stats.endpoints import teamplayeronoffdetails

SEASON = "2025-26"
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_on_off_stats():
    teams = pd.read_csv(RAW / "team_stats.csv")
    all_stats = []
    errors = []

    for i, team in teams.iterrows():
        team_id = int(team["TEAM_ID"])
        team_name = team["TEAM_NAME"]
        print(f"[{i + 1}/{len(teams)}] Fetching {team_name}...", flush=True)

        try:
            result = teamplayeronoffdetails.TeamPlayerOnOffDetails(
                team_id=team_id,
                season=SEASON,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame",
                timeout=60,
            )
            frames = result.get_data_frames()

            if len(frames) < 3:
                raise RuntimeError(f"Expected 3 tables, received {len(frames)}")

            for status, df in [("ON", frames[1]), ("OFF", frames[2])]:
                if not df.empty:
                    all_stats.append(df.assign(COURT_STATUS=status))

        except Exception as e:
            errors.append((team_name, str(e)))
            print(f"  ERROR: {e}", flush=True)

        time.sleep(1)

    if not all_stats:
        raise RuntimeError("No player on/off data was returned.")

    combined = pd.concat(all_stats, ignore_index=True)
    output = RAW / "player_on_off_stats.csv"
    combined.to_csv(output, index=False)

    print(f"\nSaved {len(combined)} rows to {output}")
    print(f"Teams successfully fetched: {combined['TEAM_ID'].nunique()}")
    print(f"Players: {combined['VS_PLAYER_ID'].nunique()}")
    print("\nRows by court status:")
    print(combined["COURT_STATUS"].value_counts())

    if errors:
        print(f"\nFailed teams: {len(errors)}")
        for team_name, error in errors:
            print(f"- {team_name}: {error}")


if __name__ == "__main__":
    fetch_on_off_stats()
