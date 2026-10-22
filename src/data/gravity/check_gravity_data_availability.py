"""Check six-factor coverage among players on NBA's 2025-26 GRAV leaderboard.

This script only reads existing CSV files. It does not change or create any datasets,
call the NBA API, standardize predictors, or fit a regression.

Suggested location: NBA-Gravity/src/data/check_gravity_data_availability.py
Run from the NBA-Gravity directory:
    python src/data/check_gravity_data_availability.py
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
SEARCH_DIRS = [HERE, HERE.parent, HERE.parent.parent]


def find_file(filename):
    project_root = HERE.parent.parent
    matches = list(project_root.rglob(filename))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Multiple copies found for {filename}: {matches}"
        )

    return None



def load_csv(filename):
    path = find_file(filename)
    if path is None:
        return None
    df = pd.read_csv(path, dtype={"PLAYER_ID": "string", "PLAYERID": "string"})
    if "PLAYER_ID" not in df and "PLAYERID" in df:
        df = df.rename(columns={"PLAYERID": "PLAYER_ID"})
    if "PLAYER_ID" not in df:
        raise ValueError(f"{filename} does not contain PLAYER_ID / PLAYERID")
    df["PLAYER_ID"] = df["PLAYER_ID"].astype("string").str.strip()
    print(f"  Loaded {filename} ({len(df)} rows)")
    return df


def player_ids_with_value(df, value_column, min_clutch_minutes=False, require_pair=False):
    if df is None:
        return None
    if value_column not in df.columns:
        raise ValueError(f"Expected column {value_column!r}; available: {df.columns.tolist()}")
    valid = pd.to_numeric(df[value_column], errors="coerce").notna()
    if require_pair and "VALID_PAIR" in df.columns:
        valid = valid & df["VALID_PAIR"].astype("string").str.lower().eq("true")
    if min_clutch_minutes:
        if "MIN" not in df.columns:
            raise ValueError("Clutch data is missing its MIN column")
        valid = valid & pd.to_numeric(df["MIN"], errors="coerce").ge(30)
    return set(df.loc[valid, "PLAYER_ID"].dropna())


def main():
    print("\nChecking coverage for official 2025-26 NBA GRAV players\n")
    gravity = load_csv("player_gravity_official.csv")
    if gravity is None:
        print("\nRequired player_gravity_official.csv was not found.")
        return

    grav_col = "GRAV" if "GRAV" in gravity.columns else "AVGGRAVITYSCORE"
    targets = gravity.loc[pd.to_numeric(gravity[grav_col], errors="coerce").notna()]
    if targets["PLAYER_ID"].duplicated().any():
        raise ValueError("Official Gravity CSV contains repeated player IDs")
    target_ids = set(targets["PLAYER_ID"].dropna())
    print(f"\nPlayers with numeric official GRAV: {len(target_ids)}\n")

    # Factor labels are in the team's agreed order. An absent file is reported,
    # not treated as if all players had missing measurements.
    factors = [
        ("1. Offensive rating ON/OFF", "player_off_rating_on_off.csv", "OFF_RATING_DIFF", False, True),
        ("2. Teammates-only eFG% ON/OFF", "player_teammate_efg_on_off.csv", "EFG_DIFF_PP", False, True),
        ("3. Assists", "player_stats.csv", "AST", False, False),
        ("4. Opponent timeout responses", "player_timeout_rates.csv", "TIMEOUTS_PER_36", False, False),
        ("5. Clutch scoring (30+ min)", "player_clutch_rates.csv", "CLUTCH_PTS_PER_36", True, False),
        ("6. Offensive Win Shares (OWS)", "player_win_shares_with_nba_ids.csv", "OWS", False, False),
    ]

    coverage = {}
    for label, filename, column, min_minutes, paired in factors:
        df = load_csv(filename)
        valid_ids = player_ids_with_value(df, column, min_minutes, paired)
        coverage[label] = valid_ids
        if valid_ids is None:
            print(f"    {label}: FILE NOT FOUND ({filename})")
        else:
            n = len(target_ids & valid_ids)
            print(f"    {label}: {n}/{len(target_ids)} GRAV players have a valid value")

    print("\n--- Summary ---")
    if all(ids is not None for ids in coverage.values()):
        complete = set.intersection(target_ids, *coverage.values())
        print(f"Players with GRAV and ALL SIX factors: {len(complete)}/{len(target_ids)}")
        for label, ids in coverage.items():
            print(f"  Missing {label}: {len(target_ids - ids)}")
    else:
        available = [ids for ids in coverage.values() if ids is not None]
        overlap = set.intersection(target_ids, *available)
        print(f"Players matching every AVAILABLE file: {len(overlap)}/{len(target_ids)}")
        print("The all-six-factors count cannot be calculated until missing files are present.")

    print("\nIMPORTANT: Coverage does not resolve multiple team stints or validate")
    print("units/aggregation. The uploaded Win Shares file contains OWS, not total WS.")
    print("No files were modified and no regression was run.")


if __name__ == "__main__":
    main()
