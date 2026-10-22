"""Combine the six 2025-26 gravity predictors with official NBA GRAV.

Usage (from NBA-Gravity root):
    python src/data/merge_gravity_model_data.py

Expected script location: NBA-Gravity/src/data/
Searches for the seven input CSVs anywhere beneath NBA-Gravity.
Writes ONLY a new player_gravity_model_data.csv beside this script.
Never changes source CSVs, imputes missing values, or fits a regression.

One row per official GRAV player (234 expected); players missing the
30-minute clutch factor are retained with an empty clutch value.
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = BASE / "player_gravity_model_data.csv"


def read_unique(filename, id_col="PLAYER_ID"):
    # Ignore a backup copy if its filename differs; raise on same-named copies.
    matches = sorted(path for path in ROOT.rglob(filename) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '{filename}' under {ROOT}; found {len(matches)}: {matches}"
        )
    df = pd.read_csv(matches[0])
    if id_col not in df.columns:
        raise ValueError(f"{filename}: expected player ID column '{id_col}'")
    if df[id_col].isna().any():
        raise ValueError(f"{filename}: missing player IDs")
    df[id_col] = pd.to_numeric(df[id_col], errors="raise").astype("int64")
    duplicate = df.loc[df[id_col].duplicated(keep=False), id_col].unique()
    if len(duplicate):
        raise ValueError(f"{filename}: duplicate IDs (sample): {duplicate[:10].tolist()}")
    print(f"Loaded {filename}: {len(df)} player rows")
    return df


def factor(filename, source_column, output_column=None):
    df = read_unique(filename)
    if source_column not in df:
        raise ValueError(f"{filename}: required column '{source_column}' not found")
    out = output_column or source_column
    result = df[["PLAYER_ID", source_column]].copy()
    result[source_column] = pd.to_numeric(result[source_column], errors="coerce")
    return result.rename(columns={source_column: out})


def main():
    gravity = read_unique("player_gravity_official.csv")
    grav_col = "GRAV" if "GRAV" in gravity else "AVGGRAVITYSCORE"
    for col in ("FIRSTNAME", "LASTNAME", grav_col):
        if col not in gravity:
            raise ValueError(f"Gravity source missing '{col}'")
    result = gravity[["PLAYER_ID", "FIRSTNAME", "LASTNAME", grav_col]].copy()
    result["PLAYER_NAME"] = (
        result["FIRSTNAME"].astype(str).str.strip() + " " +
        result["LASTNAME"].astype(str).str.strip()
    )
    result = result[["PLAYER_ID", "PLAYER_NAME", grav_col]].rename(
        columns={grav_col: "GRAV"}
    )
    result["GRAV"] = pd.to_numeric(result["GRAV"], errors="coerce")
    if result["GRAV"].isna().any():
        raise ValueError("Official GRAV source has missing/non-numeric GRAV scores")
    n = len(result)

    # The ON/OFF input is ALREADY aggregated across traded-player team stints.
    # Never join the original 701-row team-level files here.
    on_off = read_unique("player_on_off_aggregated.csv")
    for col in ("OFF_RATING_DIFF", "EFG_DIFF_PP"):
        if col not in on_off:
            raise ValueError(f"Aggregated ON/OFF file missing '{col}'")
        on_off[col] = pd.to_numeric(on_off[col], errors="coerce")
    on_off = on_off[["PLAYER_ID", "OFF_RATING_DIFF", "EFG_DIFF_PP"]]
    result = result.merge(on_off, on="PLAYER_ID", how="left", validate="one_to_one")

    factors = [
        # Factor 3 is AST PER GAME from player_stats.csv, not season totals.
        ("player_stats.csv", "AST", "AST_PER_GAME"),
        ("player_timeout_rates.csv", "TIMEOUTS_PER_36", "TIMEOUTS_PER_36"),
        ("player_clutch_rates.csv", "CLUTCH_PTS_PER_36", "CLUTCH_PTS_PER_36"),
        ("player_win_shares_with_nba_ids.csv", "OWS", "OWS"),
    ]
    for filename, col, out in factors:
        result = result.merge(
            factor(filename, col, out),
            on="PLAYER_ID", how="left", validate="one_to_one"
        )

    predictors = [
        "OFF_RATING_DIFF", "EFG_DIFF_PP", "AST_PER_GAME",
        "TIMEOUTS_PER_36", "CLUTCH_PTS_PER_36", "OWS",
    ]
    if len(result) != n or result["PLAYER_ID"].duplicated().any():
        raise AssertionError("Merge changed the number of GRAV players or duplicated a player")

    result["ALL_SIX_AVAILABLE"] = result[predictors].notna().all(axis=1)
    result = result.sort_values("GRAV", ascending=False).reset_index(drop=True)
    print("\n--- Coverage of official GRAV players ---")
    for i, col in enumerate(predictors, start=1):
        print(f"  Factor {i}: {col}: {result[col].notna().sum()}/{n}")
    complete = int(result["ALL_SIX_AVAILABLE"].sum())
    print(f"Players with all six predictors: {complete}/{n}")
    print(f"Players missing one or more predictors: {n - complete}/{n}")
    print("\nPreview (top five official GRAV players):")
    print(result.head(5).to_string(index=False))

    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Rename or move it before rerunning; "
            "existing output will not be overwritten."
        )
    result.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(result)} rows to {OUTPUT}")
    print("No source files changed, no missing values filled, no regression run.")


if __name__ == "__main__":
    main()
