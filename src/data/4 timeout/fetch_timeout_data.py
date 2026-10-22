"""NBA 2025-26 regular-season opposing timeout responses after made field goals.

Begin with TEST_GAME_ID set to the previously inspected game. After verifying
its two expected responses, set TEST_GAME_ID = None for the full season.
"""

from pathlib import Path
import re
import time

import pandas as pd
from nba_api.stats.endpoints import (
    leaguegamefinder,
    leaguedashplayerstats,
    playbyplayv3,
)
from nba_api.stats.static import teams

SEASON = "2025-26"
TEST_GAME_ID = None
PAUSE_SECONDS = 1.0
MAX_ATTEMPTS = 3

BASE_DIR = Path(__file__).resolve().parent
PROGRESS_DIR = BASE_DIR / "timeout_progress_2025_26"
OUTPUT_FILE = BASE_DIR / "player_timeout_rates.csv"
EVENT_COLUMNS = [
    "GAME_ID", "PLAYER_ID", "PLAYER_NAME", "SCORING_TEAM",
    "TIMEOUT_TEAM", "PERIOD", "SHOT_CLOCK", "TIMEOUT_CLOCK",
]


def timeout_team_aliases():
    """Match names in timeout descriptions to NBA team abbreviations."""
    aliases = {}
    for team in teams.get_teams():
        aliases[team["nickname"].casefold()] = team["abbreviation"]
        aliases[team["full_name"].casefold()] = team["abbreviation"]
        aliases[team["abbreviation"].casefold()] = team["abbreviation"]
    aliases.update({"sixers": "PHI", "blazers": "POR", "la clippers": "LAC"})
    return aliases


def parse_timeout_team(description, aliases):
    match = re.match(r"^\s*(.*?)\s+Timeout\s*:", str(description), flags=re.I)
    if match is None:
        return None
    name = " ".join(match.group(1).casefold().split())
    return aliases.get(name)


def get_regular_season_game_ids():
    finder = leaguegamefinder.LeagueGameFinder(
        player_or_team_abbreviation="T",
        season_nullable=SEASON,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
        timeout=60,
    )
    games = finder.get_data_frames()[0]
    return sorted(games["GAME_ID"].dropna().astype(str).str.zfill(10).unique())


def get_player_minutes():
    response = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Base",
        per_mode_detailed="Totals",
        timeout=60,
    )
    players = response.get_data_frames()[0]
    players = players[["PLAYER_ID", "PLAYER_NAME", "GP", "MIN"]].copy()
    return players.rename(columns={"MIN": "TOTAL_MIN"})


def analyze_game(game_id, aliases):
    response = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=60)
    actions = response.get_data_frames()[0]
    actions = actions.sort_values("actionId", kind="stable").reset_index(drop=True)

    # Reject unrecognized timeout teams rather than silently undercounting.
    for description in actions.loc[actions["actionType"] == "Timeout", "description"]:
        if parse_timeout_team(description, aliases) is None:
            raise ValueError(f"Unknown timeout team in game {game_id}: {description}")

    results = []
    for index in actions.index[actions["actionType"] == "Timeout"]:
        if index == 0:
            continue
        shot = actions.iloc[index - 1]
        timeout = actions.iloc[index]
        if shot["actionType"] != "Made Shot":
            continue
        if shot["period"] != timeout["period"]:
            continue

        scoring_team = str(shot["teamTricode"]).strip().upper()
        calling_team = parse_timeout_team(timeout["description"], aliases)
        player_id = pd.to_numeric(shot["personId"], errors="coerce")

        if scoring_team not in aliases.values() or pd.isna(player_id) or player_id <= 0:
            raise ValueError(f"Missing scorer identification in game {game_id}")
        if calling_team == scoring_team:
            continue  # A team's own timeout after its basket does not count.

        results.append({
            "GAME_ID": game_id,
            "PLAYER_ID": int(player_id),
            "PLAYER_NAME": shot["playerName"],
            "SCORING_TEAM": scoring_team,
            "TIMEOUT_TEAM": calling_team,
            "PERIOD": shot["period"],
            "SHOT_CLOCK": shot["clock"],
            "TIMEOUT_CLOCK": timeout["clock"],
        })

    return pd.DataFrame(results, columns=EVENT_COLUMNS)


def process_game(game_id, aliases):
    checkpoint = PROGRESS_DIR / f"{game_id}.csv"
    if checkpoint.exists():
        return pd.read_csv(checkpoint)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            events = analyze_game(game_id, aliases)
            # Save even a zero-event game: the header-only file marks it complete.
            temporary = checkpoint.with_suffix(".tmp")
            events.to_csv(temporary, index=False)
            temporary.replace(checkpoint)
            return events
        except ValueError:
            raise  # Data-format problems need review, not blind retries.
        except Exception as exc:
            print(f"  Attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}")
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(2 * attempt)


def main():
    PROGRESS_DIR.mkdir(exist_ok=True)
    all_game_ids = get_regular_season_game_ids()
    players = get_player_minutes()
    aliases = timeout_team_aliases()
    print(f"Regular-season games found: {len(all_game_ids)}")
    print(f"Players with season minutes: {len(players)}")

    if TEST_GAME_ID is not None:
        if TEST_GAME_ID not in all_game_ids:
            raise ValueError(f"Test game {TEST_GAME_ID} not found in season")
        selected_ids = [TEST_GAME_ID]
    else:
        selected_ids = all_game_ids

    errors = []
    consecutive_failures = 0

    for number, game_id in enumerate(selected_ids, start=1):
        try:
            already_saved = (
                PROGRESS_DIR / f"{game_id}.csv"
            ).exists()

            events = process_game(game_id, aliases)
            consecutive_failures = 0

            status = "CACHED" if already_saved else "NEW"

            print(
                f"[{number}/{len(selected_ids)}] "
                f"{game_id}: {len(events)} responses ({status})"
            )

            if TEST_GAME_ID is not None:
                print(events.to_string(index=False))

        except Exception as exc:
            errors.append(game_id)
            consecutive_failures += 1
            print(f"CHECK GAME {game_id}: {exc}")

            if consecutive_failures >= 3:
                print(
                    "\nStopping after 3 consecutive failures."
                    "\nSaved progress has been preserved."
                )
                break

        if not already_saved:
            time.sleep(PAUSE_SECONDS)


    if TEST_GAME_ID is not None:
        print("\nTest only: no final season CSV created.")
        print("Confirm two responses (Garland and Bogdanović) before full collection.")
        return

    completed_ids = [gid for gid in all_game_ids if (PROGRESS_DIR / f"{gid}.csv").exists()]
    print(f"\nCompleted games: {len(completed_ids)}/{len(all_game_ids)}")
    if errors or len(completed_ids) != len(all_game_ids):
        print("Season incomplete. No final CSV created; rerun to resume.")
        return

    event_frames = [pd.read_csv(PROGRESS_DIR / f"{gid}.csv") for gid in all_game_ids]
    all_events = pd.concat(event_frames, ignore_index=True)
    if all_events.empty:
        counts = pd.DataFrame(columns=["PLAYER_ID", "OPP_TIMEOUTS"])
    else:
        counts = all_events.groupby("PLAYER_ID").size().rename("OPP_TIMEOUTS").reset_index()

    result = players.merge(counts, on="PLAYER_ID", how="left", validate="one_to_one")
    result["OPP_TIMEOUTS"] = result["OPP_TIMEOUTS"].fillna(0).astype(int)
    result["TIMEOUTS_PER_36"] = (
        36 * result["OPP_TIMEOUTS"] / result["TOTAL_MIN"].replace(0, float("nan"))
    )
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(result)} player rows to {OUTPUT_FILE.name}")
    print(result.nlargest(10, "TIMEOUTS_PER_36").to_string(index=False))


if __name__ == "__main__":
    main()
