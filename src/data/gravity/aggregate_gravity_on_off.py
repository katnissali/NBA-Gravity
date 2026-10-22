"""Aggregate 2025-26 player/team ON/OFF factors into one row per NBA player.

Run from NBA-Gravity project root:
    python src/data/aggregate_gravity_on_off.py

READS existing raw CSVs wherever they are under project root.
WRITES one NEW summary CSV next to this script: player_on_off_aggregated.csv
No API calls, no regression, no modifications to raw CSVs.

For each factor separately, combine valid player-team differences with
ON-court MIN weights. Missing team stints have no weight; they are NOT zeroes.
This is a minutes-weighted mean of team-stint differences, not a true
season-wide on/off possession or shot-based calculation.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]  # Script is expected in src/data/
OUTPUT = BASE_DIR / "player_on_off_aggregated.csv"


def load(filename):
    matches = sorted(p for p in ROOT.rglob(filename) if p.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {filename} below {ROOT}, found {len(matches)}: {matches}"
        )
    df = pd.read_csv(matches[0])
    for col in ("PLAYER_ID", "TEAM_ID"):
        if col not in df:
            raise ValueError(f"{filename}: missing required column {col}")
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
    if df.duplicated(["PLAYER_ID", "TEAM_ID"]).any():
        raise ValueError(f"{filename}: duplicate (PLAYER_ID, TEAM_ID) pairs")
    for col in ("MIN_ON",):
        if col not in df:
            raise ValueError(f"{filename}: missing {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"Loaded {filename}: {len(df)} player-team rows")
    return df


def weighted_summary(df, value_column, prefix):
    if value_column not in df:
        raise ValueError(f"Missing expected column: {value_column}")
    tmp = df[["PLAYER_ID", "TEAM_ID", "MIN_ON", value_column]].copy()
    tmp[value_column] = pd.to_numeric(tmp[value_column], errors="coerce")
    usable = tmp[
        tmp[value_column].notna() & tmp["MIN_ON"].notna() &
        (tmp["MIN_ON"] > 0)
    ].copy()
    usable["weighted_value"] = usable[value_column] * usable["MIN_ON"]
    result = usable.groupby("PLAYER_ID", as_index=False).agg(
        weighted_sum=("weighted_value", "sum"),
        weight_sum=("MIN_ON", "sum"),
        valid_stints=("TEAM_ID", "nunique"),
    )
    result[value_column] = result["weighted_sum"] / result["weight_sum"]
    return result[["PLAYER_ID", value_column, "weight_sum", "valid_stints"]].rename(
        columns={"weight_sum": f"{prefix}_MIN_USED",
                 "valid_stints": f"{prefix}_STINTS_USED"}
    )


def main():
    off = load("player_off_rating_on_off.csv")
    efg = load("player_teammate_efg_on_off.csv")
    gravity = load_gravity()

    # Check whether the NBA source endpoints agree on each stint's ON minutes.
    minute_check = off[["PLAYER_ID", "TEAM_ID", "MIN_ON"]].merge(
        efg[["PLAYER_ID", "TEAM_ID", "MIN_ON"]],
        on=["PLAYER_ID", "TEAM_ID"], how="inner", suffixes=("_OFFRTG", "_EFG"),
        validate="one_to_one"
    )
    discrepancy = (
        minute_check["MIN_ON_OFFRTG"].notna() &
        minute_check["MIN_ON_EFG"].notna() &
        (minute_check["MIN_ON_OFFRTG"] - minute_check["MIN_ON_EFG"]).abs().gt(0.01)
    )
    if discrepancy.any():
        print(f"WARNING: {int(discrepancy.sum())} source stints have differing ON minutes; "
              "each factor will use minutes from its own source.")

    # All unique players are retained, including those with no valid ON/OFF pair.
    source_rows = pd.concat(
        [off[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"]],
         efg[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"]]],
        ignore_index=True,
    )
    names = source_rows.groupby("PLAYER_ID", as_index=False)["PLAYER_NAME"].first()
    counts = source_rows.groupby("PLAYER_ID", as_index=False)["TEAM_ID"].nunique()
    counts = counts.rename(columns={"TEAM_ID": "TEAM_STINTS_RECORDED"})

    off_summary = weighted_summary(off, "OFF_RATING_DIFF", "OFFRTG")
    efg_summary = weighted_summary(efg, "EFG_DIFF_PP", "EFG")
    result = (names.merge(counts, on="PLAYER_ID", validate="one_to_one")
              .merge(off_summary, how="left", on="PLAYER_ID", validate="one_to_one")
              .merge(efg_summary, how="left", on="PLAYER_ID", validate="one_to_one"))
    result["OFFRTG_STINTS_EXCLUDED"] = (result["TEAM_STINTS_RECORDED"] -
                                         result["OFFRTG_STINTS_USED"].fillna(0)).astype(int)
    result["EFG_STINTS_EXCLUDED"] = (result["TEAM_STINTS_RECORDED"] -
                                      result["EFG_STINTS_USED"].fillna(0)).astype(int)
    result = result.sort_values("PLAYER_ID").reset_index(drop=True)
    if result["PLAYER_ID"].duplicated().any():
        raise AssertionError("Aggregation did not yield one record per player")

    print(f"\nUnique player rows: {len(result)}")
    print(f"Players with multiple recorded teams: {(result['TEAM_STINTS_RECORDED'] > 1).sum()}")
    print(f"Players with valid aggregated OffRtg: {result['OFF_RATING_DIFF'].notna().sum()}")
    print(f"Players with valid aggregated teammate eFG: {result['EFG_DIFF_PP'].notna().sum()}")

    if gravity is not None:
        matched = result[result["PLAYER_ID"].isin(gravity)].copy()
        print(f"Official GRAV players matched: {len(matched)}/{len(gravity)}")
        print("GRAV players with both aggregated factors:",
              int(matched[["OFF_RATING_DIFF", "EFG_DIFF_PP"]].notna().all(axis=1).sum()))
        print("GRAV players with multiple recorded teams:",
              int((matched["TEAM_STINTS_RECORDED"] > 1).sum()))

    for pid, name in [(201935, "James Harden"), (1628379, "Luke Kennard"),
                      (1626145, "Tyus Jones")]:
        sample = result.loc[result["PLAYER_ID"].eq(pid)]
        if not sample.empty:
            s = sample.iloc[0]
            print(f"\n{name}: OffRtg diff = {s['OFF_RATING_DIFF']:.4f} pts/100 possessions; "
                  f"teammate eFG diff = {s['EFG_DIFF_PP']:.4f} percentage points")
            print(f"  OFFRTG stints used = {int(s['OFFRTG_STINTS_USED'])}, "
                  f"EFG stints used = {int(s['EFG_STINTS_USED'])}, "
                  f"recorded = {int(s['TEAM_STINTS_RECORDED'])}")

    result.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(result)} rows to {OUTPUT}")
    print("Original CSVs unchanged. No regression run.")


def load_gravity():
    paths = sorted(p for p in ROOT.rglob("player_gravity_official.csv") if p.is_file())
    if len(paths) != 1:
        print("Official GRAV file not uniquely found: skipping GRAV coverage check")
        return None
    df = pd.read_csv(paths[0])
    idcol = "PLAYER_ID" if "PLAYER_ID" in df else "PLAYERID"
    return set(pd.to_numeric(df[idcol], errors="raise").astype(int))


if __name__ == "__main__":
    main()
