"""Collect 2025-26 NBA team offensive rating ON/OFF for every player-team stint.

Requires: pip install nba_api pandas
Run: python fetch_off_rating_data.py

The NBA reports OFF_RATING directly as team points scored per 100 possessions.
The difference is ON - OFF. This is a team-level on/off metric, NOT a
teammate-only metric, and not a causal estimate of a player's impact.

Each team is cached after a successful request so interrupted runs can resume.
Players who appeared on more than one team retain one row per team; don't
merge this file one-to-one against season-level player data until the stints
have been appropriately handled.
"""

from pathlib import Path
import time

import pandas as pd
from nba_api.stats.endpoints import teamplayeronoffsummary
from nba_api.stats.static import teams


SEASON = "2025-26"
REQUEST_TIMEOUT_SECONDS = 60
PAUSE_SECONDS = 1.5
RETRY_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 8
MAX_CONSECUTIVE_FAILURES = 3

# Save relative to this script, not the directory from which Python is run.
BASE_DIR = Path(__file__).resolve().parent
PROGRESS_DIR = BASE_DIR / "off_rating_progress_2025_26"
OUTPUT_FILE = BASE_DIR / "player_off_rating_on_off.csv"


ON_COLUMNS = {
    "VS_PLAYER_ID": "PLAYER_ID",
    "VS_PLAYER_NAME": "NAME_ON",
    "GP": "GP_ON",
    "MIN": "MIN_ON",
    "OFF_RATING": "OFF_RATING_ON",
}
OFF_COLUMNS = {
    "VS_PLAYER_ID": "PLAYER_ID",
    "VS_PLAYER_NAME": "NAME_OFF",
    "GP": "GP_OFF",
    "MIN": "MIN_OFF",
    "OFF_RATING": "OFF_RATING_OFF",
}
OUTPUT_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "GP_ON",
    "MIN_ON",
    "OFF_RATING_ON",
    "GP_OFF",
    "MIN_OFF",
    "OFF_RATING_OFF",
    "OFF_RATING_DIFF",
    "VALID_PAIR",
]


def subset_and_rename(frame: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    missing = set(mapping).difference(frame.columns)
    if missing:
        raise ValueError(f"Unexpected NBA API columns; missing: {sorted(missing)}")
    result = frame[list(mapping)].rename(columns=mapping).copy()
    result["PLAYER_ID"] = pd.to_numeric(result["PLAYER_ID"], errors="raise").astype("int64")
    if result["PLAYER_ID"].duplicated().any():
        raise ValueError("Repeated player IDs in one team's ON or OFF response")
    return result


def download_team(team: dict) -> pd.DataFrame:
    api = teamplayeronoffsummary.TeamPlayerOnOffSummary(
        team_id=team["id"],
        season=SEASON,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Totals",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    on_raw = api.players_on_court_team_player_on_off_summary.get_data_frame()
    off_raw = api.players_off_court_team_player_on_off_summary.get_data_frame()
    if on_raw.empty or off_raw.empty:
        raise ValueError("NBA API returned an empty ON or OFF table")

    on = subset_and_rename(on_raw, ON_COLUMNS)
    off = subset_and_rename(off_raw, OFF_COLUMNS)
    merged = on.merge(off, on="PLAYER_ID", how="outer", validate="one_to_one")

    merged["PLAYER_NAME"] = merged["NAME_ON"].combine_first(merged["NAME_OFF"])
    for col in (
        "GP_ON", "MIN_ON", "OFF_RATING_ON",
        "GP_OFF", "MIN_OFF", "OFF_RATING_OFF",
    ):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["OFF_RATING_DIFF"] = (
        merged["OFF_RATING_ON"] - merged["OFF_RATING_OFF"]
    )
    merged["VALID_PAIR"] = (
        merged["OFF_RATING_ON"].notna() & merged["OFF_RATING_OFF"].notna()
    )
    if not merged["VALID_PAIR"].any():
        raise ValueError("No players have both numeric ON and OFF ratings")

    merged["TEAM_ID"] = team["id"]
    merged["TEAM_ABBREVIATION"] = team["abbreviation"]
    merged["TEAM_NAME"] = team["full_name"]
    return merged[OUTPUT_COLUMNS].sort_values("PLAYER_ID").reset_index(drop=True)


def get_team_data(team: dict) -> tuple[pd.DataFrame, bool]:
    checkpoint = PROGRESS_DIR / f"{team['id']}.csv"
    if checkpoint.exists():
        cached = pd.read_csv(checkpoint)
        if not set(OUTPUT_COLUMNS).issubset(cached.columns):
            raise ValueError(f"Checkpoint has unexpected columns: {checkpoint}")
        return cached[OUTPUT_COLUMNS], True

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            result = download_team(team)
            result.to_csv(checkpoint, index=False)
            return result, False
        except Exception as exc:
            print(f"  Attempt {attempt}/{RETRY_ATTEMPTS} failed: {exc}", flush=True)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS)
    raise RuntimeError(f"Failed to collect {team['abbreviation']} after {RETRY_ATTEMPTS} attempts")


def main() -> None:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    nba_teams = sorted(teams.get_teams(), key=lambda t: t["abbreviation"])
    print(f"Collecting {SEASON} team OffRtg ON/OFF for {len(nba_teams)} teams", flush=True)

    all_results = []
    errors = []
    consecutive_failures = 0

    for number, team in enumerate(nba_teams, start=1):
        was_cached = False
        try:
            frame, was_cached = get_team_data(team)
            all_results.append(frame)
            consecutive_failures = 0
            status = "CACHED" if was_cached else "NEW"
            print(
                f"[{number}/{len(nba_teams)}] {team['abbreviation']}: "
                f"{len(frame)} player-team rows ({status})",
                flush=True,
            )
        except Exception as exc:
            errors.append(team["abbreviation"])
            consecutive_failures += 1
            print(f"CHECK TEAM {team['abbreviation']}: {exc}", flush=True)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("Stopping after 3 consecutive team failures. Cached teams are safe.")
                break

        if not was_cached:
            time.sleep(PAUSE_SECONDS)

    print(f"\nCompleted teams: {len(all_results)}/{len(nba_teams)}")
    if len(all_results) != len(nba_teams):
        print("No new final CSV created: some teams are missing.")
        print("Failed teams:", ", ".join(errors))
        print("Rerun the same command to resume from cached teams.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    if combined.duplicated(subset=["PLAYER_ID", "TEAM_ID"]).any():
        raise ValueError("Unexpected duplicate PLAYER_ID + TEAM_ID rows")

    combined = combined.sort_values(["PLAYER_ID", "TEAM_ID"]).reset_index(drop=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    unique_players = combined["PLAYER_ID"].nunique()
    traded_players = (combined.groupby("PLAYER_ID")["TEAM_ID"].nunique() > 1).sum()
    valid_pairs = int(combined["VALID_PAIR"].fillna(False).astype(bool).sum())
    print(f"Saved {len(combined)} player-team rows to {OUTPUT_FILE.name}")
    print(f"Unique players: {unique_players}")
    print(f"Players with multiple teams: {traded_players}")
    print(f"Rows with both ON and OFF ratings: {valid_pairs}")

    curry = combined[combined["PLAYER_ID"] == 201939]
    if not curry.empty:
        print("\nStephen Curry verification:")
        print(
            curry[["PLAYER_NAME", "TEAM_ABBREVIATION", "OFF_RATING_ON",
                   "OFF_RATING_OFF", "OFF_RATING_DIFF"]].to_string(index=False)
        )


if __name__ == "__main__":
    main()
