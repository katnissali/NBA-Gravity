"""Collect 2025-26 NBA *teammates-only* eFG% ON/OFF, one row per player-team.

Requires: pip install pandas nba_api
From the NBA-Gravity project folder:
    python src/data/fetch_teammate_efg_data.py

Definition for focal player i:
    eFG_teammates_ON = ((team_ON_FGM - player_FGM)
                        + 0.5*(team_ON_FG3M - player_FG3M))
                       / (team_ON_FGA - player_FGA)
    eFG_teammates_OFF = (team_OFF_FGM + 0.5*team_OFF_FG3M) / team_OFF_FGA
    EFG_DIFF_PP = 100*(eFG_teammates_ON - eFG_teammates_OFF)

All totals come from 2025-26 regular-season, Totals mode, Base measure.
ON = same-team shooting during player's on-court minutes, excluding focal player.
OFF = team shooting while focal player is off court (already excludes player).
This measures an association, not causality. Team composition/opponents may differ.

Traded players appear once per team: do NOT naively average stint percentages.
Raw makes/attempts are preserved so player-season pooling is possible later.
Successful team results are cached relative to this script; reruns reuse them.
"""

from pathlib import Path
import time

import pandas as pd
from nba_api.stats.endpoints import teamplayeronoffdetails, teamplayerdashboard
from nba_api.stats.static import teams

SEASON = "2025-26"
REQUEST_TIMEOUT_SECONDS = 60
PAUSE_SECONDS = 1.5
RETRY_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 8
MAX_CONSECUTIVE_FAILURES = 3

BASE_DIR = Path(__file__).resolve().parent
PROGRESS_DIR = BASE_DIR / "teammate_efg_progress_2025_26"
OUTPUT_FILE = BASE_DIR / "player_teammate_efg_on_off.csv"

# Both APIs use NBA-provided total counts. Do not use rounded percentage columns.
ON_COLUMNS = {
    "VS_PLAYER_ID": "PLAYER_ID",
    "VS_PLAYER_NAME": "NAME_ON",
    "GP": "GP_ON",
    "MIN": "MIN_ON",
    "FGM": "FGM_TEAM_ON",
    "FGA": "FGA_TEAM_ON",
    "FG3M": "FG3M_TEAM_ON",
}
OFF_COLUMNS = {
    "VS_PLAYER_ID": "PLAYER_ID",
    "VS_PLAYER_NAME": "NAME_OFF",
    "GP": "GP_OFF",
    "MIN": "MIN_OFF",
    "FGM": "FGM_TEAM_OFF",
    "FGA": "FGA_TEAM_OFF",
    "FG3M": "FG3M_TEAM_OFF",
}
PLAYER_COLUMNS = {
    "PLAYER_ID": "PLAYER_ID",
    "PLAYER_NAME": "NAME_PLAYER",
    "GP": "GP_PLAYER",
    "MIN": "MIN_PLAYER",
    "FGM": "FGM_PLAYER",
    "FGA": "FGA_PLAYER",
    "FG3M": "FG3M_PLAYER",
}

OUTPUT_COLUMNS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
    "GP_ON", "MIN_ON", "GP_OFF", "MIN_OFF", "GP_PLAYER", "MIN_PLAYER",
    "FGM_TEAM_ON", "FGA_TEAM_ON", "FG3M_TEAM_ON",
    "FGM_PLAYER", "FGA_PLAYER", "FG3M_PLAYER",
    "FGM_TEAMMATES_ON", "FGA_TEAMMATES_ON", "FG3M_TEAMMATES_ON",
    "FGM_TEAM_OFF", "FGA_TEAM_OFF", "FG3M_TEAM_OFF",
    "EFG_TEAMMATES_ON_PCT", "EFG_TEAMMATES_OFF_PCT", "EFG_DIFF_PP", "VALID_PAIR",
]


def select_columns(frame: pd.DataFrame, mapping: dict, source: str) -> pd.DataFrame:
    missing = set(mapping) - set(frame.columns)
    if missing:
        raise ValueError(f"{source}: expected columns missing: {sorted(missing)}")
    result = frame[list(mapping)].rename(columns=mapping).copy()
    result["PLAYER_ID"] = pd.to_numeric(result["PLAYER_ID"], errors="raise").astype("int64")
    if result["PLAYER_ID"].duplicated().any():
        raise ValueError(f"{source}: duplicate PLAYER_ID rows in a team response")
    for col in result.columns:
        if col not in ("PLAYER_ID", "NAME_ON", "NAME_OFF", "NAME_PLAYER"):
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def efg_pct(fgm: pd.Series, fga: pd.Series, fg3m: pd.Series) -> pd.Series:
    """Percentage scale (e.g. 57.599), NaN if denominator invalid/missing."""
    denominator = fga.where(fga > 0)
    return 100 * (fgm + 0.5 * fg3m) / denominator


def validate_shooting_counts(frame: pd.DataFrame) -> None:
    """Reject contradictions from data mismatch; missing values remain missing."""
    for label in ("TEAM_ON", "TEAM_OFF", "PLAYER", "TEAMMATES_ON"):
        fgm = frame[f"FGM_{label}"]
        fga = frame[f"FGA_{label}"]
        fg3m = frame[f"FG3M_{label}"]
        # Fields are nonnegative; 3-point FGM cannot exceed total FGM,
        # and total FGM cannot exceed attempts.
        bad = ((fgm < 0) | (fga < 0) | (fg3m < 0)
               | (fgm > fga) | (fg3m > fgm))
        if bad.any():
            ids = frame.loc[bad, "PLAYER_ID"].head(5).tolist()
            raise ValueError(f"Invalid {label} shooting totals for players: {ids}")


def download_team(team: dict) -> pd.DataFrame:
    kwargs = dict(
        team_id=team["id"],
        season=SEASON,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Base",
        per_mode_detailed="Totals",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    details = teamplayeronoffdetails.TeamPlayerOnOffDetails(**kwargs)
    on_raw = details.players_on_court_team_player_on_off_details.get_data_frame()
    off_raw = details.players_off_court_team_player_on_off_details.get_data_frame()

    dashboard = teamplayerdashboard.TeamPlayerDashboard(**kwargs)
    player_raw = dashboard.players_season_totals.get_data_frame()

    if on_raw.empty or off_raw.empty or player_raw.empty:
        raise ValueError("NBA API returned an empty ON, OFF, or player shooting table")

    on = select_columns(on_raw, ON_COLUMNS, "ON")
    off = select_columns(off_raw, OFF_COLUMNS, "OFF")
    players = select_columns(player_raw, PLAYER_COLUMNS, "individual player")

    # Preserve unmatched ON and OFF rows. Do not interpret missing rows as zero.
    combined = on.merge(off, how="outer", on="PLAYER_ID", validate="one_to_one")
    combined = combined.merge(players, how="left", on="PLAYER_ID", validate="one_to_one")
    combined["PLAYER_NAME"] = (
        combined["NAME_ON"]
        .combine_first(combined["NAME_OFF"])
        .combine_first(combined["NAME_PLAYER"])
    )

    # Player totals are restricted to this team via TeamPlayerDashboard.
    for stat in ("FGM", "FGA", "FG3M"):
        combined[f"{stat}_TEAMMATES_ON"] = (
            combined[f"{stat}_TEAM_ON"] - combined[f"{stat}_PLAYER"]
        )

    validate_shooting_counts(combined)
    combined["EFG_TEAMMATES_ON_PCT"] = efg_pct(
        combined["FGM_TEAMMATES_ON"],
        combined["FGA_TEAMMATES_ON"],
        combined["FG3M_TEAMMATES_ON"],
    )
    combined["EFG_TEAMMATES_OFF_PCT"] = efg_pct(
        combined["FGM_TEAM_OFF"],
        combined["FGA_TEAM_OFF"],
        combined["FG3M_TEAM_OFF"],
    )
    combined["EFG_DIFF_PP"] = (
        combined["EFG_TEAMMATES_ON_PCT"] - combined["EFG_TEAMMATES_OFF_PCT"]
    )
    combined["VALID_PAIR"] = combined["EFG_DIFF_PP"].notna()

    if not combined["VALID_PAIR"].any():
        raise ValueError("No valid teammates-only eFG ON/OFF comparisons on team")

    combined["TEAM_ID"] = team["id"]
    combined["TEAM_ABBREVIATION"] = team["abbreviation"]
    combined["TEAM_NAME"] = team["full_name"]
    return combined[OUTPUT_COLUMNS].sort_values("PLAYER_ID").reset_index(drop=True)


def collect_or_load_team(team: dict) -> tuple[pd.DataFrame, bool]:
    checkpoint = PROGRESS_DIR / f"{team['id']}.csv"
    if checkpoint.exists():
        cached = pd.read_csv(checkpoint)
        if not set(OUTPUT_COLUMNS).issubset(cached.columns):
            raise ValueError(f"Cache missing expected columns: {checkpoint}")
        return cached[OUTPUT_COLUMNS], True

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            result = download_team(team)
            temporary = checkpoint.with_suffix(".tmp")
            result.to_csv(temporary, index=False)
            temporary.replace(checkpoint)
            return result, False
        except ValueError:
            # A schema/data inconsistency needs investigation, not blind retries.
            raise
        except Exception as exc:
            print(f"  Attempt {attempt}/{RETRY_ATTEMPTS} failed: {exc}", flush=True)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError(f"Could not collect {team['abbreviation']} after retries")


def main() -> None:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    nba_teams = sorted(teams.get_teams(), key=lambda t: t["abbreviation"])
    print(f"Collecting {SEASON} teammates-only eFG% ON/OFF for {len(nba_teams)} teams", flush=True)

    frames = []
    failed = []
    consecutive_failures = 0
    for index, team in enumerate(nba_teams, 1):
        cached = False
        try:
            frame, cached = collect_or_load_team(team)
            frames.append(frame)
            consecutive_failures = 0
            n_valid = int(frame["VALID_PAIR"].fillna(False).astype(bool).sum())
            print(
                f"[{index}/{len(nba_teams)}] {team['abbreviation']}: "
                f"{len(frame)} player-team rows, {n_valid} valid pairs "
                f"({'CACHED' if cached else 'NEW'})",
                flush=True,
            )
        except Exception as exc:
            failed.append(team["abbreviation"])
            consecutive_failures += 1
            print(f"CHECK TEAM {team['abbreviation']}: {exc}", flush=True)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("Stopping after three consecutive failures; checkpoints are safe.")
                break
        if not cached:
            time.sleep(PAUSE_SECONDS)

    print(f"\nCompleted teams: {len(frames)}/{len(nba_teams)}")
    if len(frames) != len(nba_teams):
        print("No new final CSV created because the season is incomplete.")
        print("Failed teams:", ", ".join(failed))
        print("Run the same command again to resume from cached teams.")
        return

    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["PLAYER_ID", "TEAM_ID"]).any():
        raise ValueError("Duplicate player-team combination in combined file")
    result = result.sort_values(["PLAYER_ID", "TEAM_ID"]).reset_index(drop=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(result)} player-team rows to {OUTPUT_FILE.name}")
    print(f"Unique players: {result['PLAYER_ID'].nunique()}")
    traded = (result.groupby("PLAYER_ID")["TEAM_ID"].nunique() > 1).sum()
    print(f"Players with multiple teams: {traded}")
    print(f"Rows with valid ON/OFF teammate eFG: {int(result['VALID_PAIR'].sum())}")

    curry = result.loc[result["PLAYER_ID"] == 201939]
    if not curry.empty:
        print("\nStephen Curry verification (expected approximately 57.5991%, 53.3820%, +4.2170 pp):")
        print(curry[["PLAYER_NAME", "TEAM_ABBREVIATION", "EFG_TEAMMATES_ON_PCT",
                     "EFG_TEAMMATES_OFF_PCT", "EFG_DIFF_PP"]].to_string(index=False))


if __name__ == "__main__":
    main()
