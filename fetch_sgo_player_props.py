#!/usr/bin/env python3
"""
Fetch up to 10 NBA player props from SportsGameOdds, with robust fallbacks and debug output.

Setup
- Export your key:
    Linux/macOS:  export SGO_API_KEY=YOUR_TOKEN
    Windows PS:   setx SGO_API_KEY YOUR_TOKEN

Run
    python fetch_sgo_player_props.py --limit 10 --bookmaker fanduel --debug

Notes
- Uses header X-Api-Key and trailing-slash endpoints.
- Tries multiple fetch variants in this order until it finds events, with pagination attempts:
    1) /v2/events/ with leagueID=NBA, marketOddsAvailable=true, status=upcoming
    2) + expand=markets,bookmakers,outcomes
    3) leagueID=NBA only
    4) no params (broadest)
- If events do not include markets, fetch per-event markets via:
    - /v2/events/{eventID}/markets/?expand=bookmakers,outcomes
    - fallback: /v2/markets/?eventID=...&expand=bookmakers,outcomes (also tries eventId)
- Filters bookmaker by substring (default "fanduel"); pass --bookmaker "" to accept all.
- Add --dump-raw to print a trimmed sample payload to help diagnose structure changes.
"""

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

import requests

SGO_BASE = "https://api.sportsgameodds.com/v2"
EVENTS_URL = f"{SGO_BASE}/events/"
EVENT_MARKETS_URL_TMPL = f"{SGO_BASE}/events/{{eventID}}/markets/"
MARKETS_URL = f"{SGO_BASE}/markets/"

DEFAULT_LEAGUE_ID = "NBA"
DEFAULT_LIMIT = 10
EVENTS_PAGE_LIMIT = 100
TIMEOUT = 20
SLEEP_BETWEEN_PAGES = 0.2
RETRY_SLEEP = 0.5


def headers(api_key: Optional[str]) -> Dict[str, str]:
    if not api_key:
        print("ERROR: Set SGO_API_KEY (or SPORTSGAMEODDS_API_KEY).", file=sys.stderr)
        sys.exit(1)
    return {"X-Api-Key": api_key}


def to_int_american(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        pass
    try:
        dec = float(v)
        if dec <= 1.0:
            return None
        if dec >= 2.0:
            return int(round((dec - 1.0) * 100))
        return int(round(-100.0 / (dec - 1.0)))
    except Exception:
        return None


def is_bookmaker_match(bm_node: Dict[str, Any], substr: str) -> bool:
    if substr is None:
        return True
    s = substr.strip().lower()
    if s == "":
        return True

    # direct fields
    for k in ("name", "bookmaker", "bookmakerName", "bookmakerId", "bookmakerID",
              "id", "provider", "providerKey", "sportsbook", "source", "book", "key", "code"):
        v = bm_node.get(k)
        if v and s in str(v).lower():
            return True

    # nested bookmaker objects
    for nest in ("bookmaker", "sportsbook", "provider"):
        obj = bm_node.get(nest)
        if isinstance(obj, dict):
            for k in ("name", "id", "key", "code"):
                v = obj.get(k)
                if v and s in str(v).lower():
                    return True

    # tolerant aliases for FanDuel
    if s in ("fanduel", "fan duel", "fd"):
        for k in ("name", "bookmaker", "bookmakerName", "provider", "source", "key", "code"):
            v = bm_node.get(k)
            if v and any(tok in str(v).lower() for tok in ("fanduel", "fan duel", "fd")):
                return True
        obj = bm_node.get("bookmaker") or bm_node.get("provider")
        if isinstance(obj, dict):
            v = obj.get("name") or obj.get("key") or obj.get("code")
            if v and any(tok in str(v).lower() for tok in ("fanduel", "fan duel", "fd")):
                return True

    return False


def market_label(market_name: str) -> str:
    m = (market_name or "").lower()
    if "assist" in m:
        return "assists"
    if "rebound" in m:
        return "rebounds"
    if "3" in m or "three" in m or "3pt" in m or "3-pointer" in m or "three-pointer" in m:
        return "threes"
    if "point" in m or "pts" in m or "score" in m:
        return "points"
    return "unknown"


def try_get(d: Dict[str, Any], keys: Tuple[str, ...]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def fetch_events(api_key: str, league_id: str, debug: bool) -> List[Dict[str, Any]]:
    # Try multiple param shapes for league and optional status
    base_variants = [
        {"leagueID": league_id, "marketOddsAvailable": "true", "status": "upcoming", "limit": str(EVENTS_PAGE_LIMIT)},
        {"leagueID": league_id, "marketOddsAvailable": "true", "limit": str(EVENTS_PAGE_LIMIT), "expand": "markets,bookmakers,outcomes"},
        {"leagueId": league_id, "marketOddsAvailable": "true", "limit": str(EVENTS_PAGE_LIMIT)},
        {"league": league_id, "limit": str(EVENTS_PAGE_LIMIT)},
        {"limit": str(EVENTS_PAGE_LIMIT)},
    ]

    all_events: List[Dict[str, Any]] = []
    for var in base_variants:
        # attempt naive call first
        if debug:
            print(f"[DEBUG] GET {EVENTS_URL} params={var}")
        try:
            r = requests.get(EVENTS_URL, headers=headers(api_key), params=var, timeout=TIMEOUT)
        except Exception as e:
            if debug:
                print(f"[DEBUG] request error: {e}")
            continue

        if r.status_code == 429:
            if debug:
                print("[DEBUG] 429 Too Many Requests; sleeping then retrying once...")
            time.sleep(RETRY_SLEEP)
            r = requests.get(EVENTS_URL, headers=headers(api_key), params=var, timeout=TIMEOUT)

        if r.status_code >= 400:
            if debug:
                print(f"[DEBUG] HTTP {r.status_code} {r.reason}: {r.text[:400]}")
            continue

        data = r.json()
        events = data.get("events") or data.get("data") or []
        if debug:
            print(f"[DEBUG] events returned (first page): {len(events)}")

        # collect
        all_events.extend(events)

        # Try pagination if we got a full page
        if len(events) >= EVENTS_PAGE_LIMIT:
            # common patterns: page or offset
            for page in range(2, 4):  # fetch a couple more pages opportunistically
                paged_params = dict(var)
                paged_params["page"] = str(page)
                if debug:
                    print(f"[DEBUG] paging GET {EVENTS_URL} params={paged_params}")
                try:
                    r2 = requests.get(EVENTS_URL, headers=headers(api_key), params=paged_params, timeout=TIMEOUT)
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] paging error: {e}")
                    break
                if r2.status_code >= 400:
                    if debug:
                        print(f"[DEBUG] paging HTTP {r2.status_code}: {r2.text[:200]}")
                    break
                d2 = r2.json()
                ev2 = d2.get("events") or d2.get("data") or []
                if not ev2:
                    break
                all_events.extend(ev2)
                time.sleep(SLEEP_BETWEEN_PAGES)

        # Stop at first variant that returns anything
        if all_events:
            break

        time.sleep(0.05)

    return all_events


def fetch_markets_for_event(api_key: str, event_id: str, debug: bool) -> List[Dict[str, Any]]:
    # Prefer event-scoped markets endpoint
    u = EVENT_MARKETS_URL_TMPL.format(eventID=event_id)
    for expand in ("bookmakers,outcomes", "bookmakers,lines,outcomes", "markets,bookmakers,outcomes"):
        try:
            if debug:
                print(f"[DEBUG] GET {u} expand={expand}")
            r = requests.get(u, headers=headers(api_key), params={"expand": expand}, timeout=TIMEOUT)
            if r.status_code < 400:
                data = r.json()
                mkts = data.get("markets") or data.get("data") or []
                if isinstance(mkts, list) and mkts:
                    return mkts
            elif debug:
                print(f"[DEBUG] HTTP {r.status_code} {r.reason}: {r.text[:300]}")
        except Exception as e:
            if debug:
                print(f"[DEBUG] markets fetch error (event): {e}")

    # Fallback: markets collection filtered by eventID or eventId
    for key in ("eventID", "eventId"):
        try:
            params = {key: event_id, "expand": "bookmakers,outcomes"}
            if debug:
                print(f"[DEBUG] GET {MARKETS_URL} {key}={event_id} expand=bookmakers,outcomes")
            r = requests.get(MARKETS_URL, headers=headers(api_key), params=params, timeout=TIMEOUT)
            if r.status_code < 400:
                data = r.json()
                mkts = data.get("markets") or data.get("data") or []
                if isinstance(mkts, list):
                    return mkts
            elif debug:
                print(f"[DEBUG] HTTP {r.status_code} {r.reason}: {r.text[:300]}")
        except Exception as e:
            if debug:
                print(f"[DEBUG] markets fetch error (collection {key}): {e}")
    return []


def collect_bookmaker_names(markets: List[Dict[str, Any]]) -> List[str]:
    names = set()
    for m in markets or []:
        for bm in (m.get("bookmakers") or []):
            fields = []
            for k in ("name", "bookmaker", "bookmakerName", "bookmakerId", "bookmakerID", "id", "provider", "providerKey", "sportsbook", "source", "book", "key", "code"):
                v = bm.get(k)
                if v:
                    fields.append(str(v))
            nested = bm.get("bookmaker") or bm.get("provider")
            if isinstance(nested, dict):
                for k in ("name", "id", "key", "code"):
                    v = nested.get(k)
                    if v:
                        fields.append(str(v))
            if fields:
                names.add(" | ".join(fields))
    return sorted(names)


def extract_player_props_from_event(event: Dict[str, Any], api_key: str, bookmaker_substr: str, debug: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    markets = event.get("markets") or []
    if not markets:
        ev_id = str(try_get(event, ("eventID", "eventId", "id")) or "")
        if not ev_id:
            return out
        markets = fetch_markets_for_event(api_key, ev_id, debug)

    if debug and not markets:
        print("[DEBUG] No markets found for event; skipping.")
        return out

    home = try_get(event, ("homeTeam", "home")) or ""
    away = try_get(event, ("awayTeam", "away")) or ""
    game_label = f"{away} at {home}".strip()
    start_time = try_get(event, ("startTime", "commenceTime", "start", "commence")) or ""

    for m in markets:
        bet_type = str(try_get(m, ("betTypeID", "type", "key")) or "").lower()
        market_name = str(try_get(m, ("marketName", "name", "label")) or "")

        looks_player = ("player" in market_name.lower())
        is_ou = bet_type in ("ou", "over_under", "totals", "player_ou", "player_totals")

        if not (looks_player or is_ou):
            continue

        bms = m.get("bookmakers") or []
        for bm in bms:
            if not is_bookmaker_match(bm, bookmaker_substr):
                continue

            # outcomes may appear directly or under a 'lines' node
            outcomes = bm.get("outcomes") or []
            if not outcomes and isinstance(bm.get("lines"), list) and bm["lines"]:
                # take all outcomes from line entries
                for ln in bm["lines"]:
                    if isinstance(ln, dict) and isinstance(ln.get("outcomes"), list):
                        outcomes.extend(ln["outcomes"])

            bucket: Dict[str, Dict[str, Any]] = {}
            for oc in outcomes:
                # Side detection
                side_raw = str(try_get(oc, ("sideID", "side", "label", "name", "betSide")) or "").lower()
                side_raw = side_raw.strip()
                if side_raw in ("o", "over", "overpoints", "over_p"):
                    side = "over"
                elif side_raw in ("u", "under", "underpoints", "under_p"):
                    side = "under"
                else:
                    if side_raw.startswith("over"):
                        side = "over"
                    elif side_raw.startswith("under"):
                        side = "under"
                    else:
                        # sometimes outcome name is the player, and line is separate — skip if not OU
                        continue

                # Player
                player = try_get(oc, ("participant", "player", "runnerName", "selectionName", "name", "competitor", "athlete"))
                if not player:
                    continue

                # Line/threshold
                line = try_get(oc, ("line", "threshold", "value", "points", "total", "handicap")) or try_get(m, ("line", "threshold", "value"))
                try:
                    line = float(line)
                except Exception:
                    continue

                # Odds
                american = to_int_american(try_get(oc, ("americanOdds", "oddsAmerican", "priceAmerican", "american", "odds", "price")))
                if american is None:
                    continue

                key = f"{player}|{market_label(market_name)}|{line}"
                if key not in bucket:
                    bucket[key] = {
                        "game": game_label,
                        "game_start": start_time,
                        "market": market_label(market_name),
                        "player": str(player),
                        "line": float(line),
                    }
                if side == "over":
                    bucket[key]["odds_over"] = int(american)
                elif side == "under":
                    bucket[key]["odds_under"] = int(american)

            for rec in bucket.values():
                if "odds_over" in rec or "odds_under" in rec:
                    out.append(rec)

    # If nothing collected for this event and we filtered to FanDuel, show what we did see
    if not out and bookmaker_substr:
        names = collect_bookmaker_names(markets)
        if debug and names:
            print(f"[DEBUG] No matching bookmaker '{bookmaker_substr}'. Available bookmaker fields for this event:")
            for n in names[:12]:
                print(f"        - {n}")

    return out


def main():
    parser = argparse.ArgumentParser(description="Fetch NBA player props from SportsGameOdds.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of props to return (default 10)")
    parser.add_argument("--bookmaker", type=str, default="fanduel", help='Bookmaker substring filter ("" to accept any)')
    parser.add_argument("--league", type=str, default=DEFAULT_LEAGUE_ID, help="League identifier (default NBA)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--dump-raw", action="store_true", help="Dump a trimmed raw JSON sample for debugging")
    args = parser.parse_args()

    api_key = os.getenv("SGO_API_KEY") or os.getenv("SPORTSGAMEODDS_API_KEY")
    if not api_key:
        print("ERROR: Set SGO_API_KEY (or SPORTSGAMEODDS_API_KEY).", file=sys.stderr)
        sys.exit(1)

    events = fetch_events(api_key, args.league, args.debug)
    if not events:
        print("No events found (or API rejected parameters). Try --debug and/or --bookmaker \"\" to widen search.", file=sys.stderr)
        sys.exit(2)

    if args.debug:
        with_mkts = sum(1 for e in events if isinstance(e.get("markets"), list) and e["markets"])
        print(f"[DEBUG] Events total: {len(events)}; with pre-expanded markets: {with_mkts}")

    # Optional dump to understand the schema we got back
    if args.dump_raw:
        sample = events[0]
        ev_preview = {k: sample.get(k) for k in ("eventID", "eventId", "id", "homeTeam", "awayTeam", "startTime", "commenceTime")}
        ev_preview["markets_count"] = len(sample.get("markets") or [])
        # Try fetch markets for first event if none embedded
        if not sample.get("markets"):
            ev_id = str(try_get(sample, ("eventID", "eventId", "id")) or "")
            if ev_id:
                mkts = fetch_markets_for_event(api_key, ev_id, args.debug)
                ev_preview["fetched_markets_count"] = len(mkts)
                # include just first couple bookmakers to reduce noise
                if mkts:
                    m0 = mkts[0]
                    ev_preview["sample_market"] = {
                        "name": try_get(m0, ("marketName", "name", "label")),
                        "bookmakers_preview": [
                            { "name_fields": [str(v) for v in [
                                bm.get("name"), bm.get("bookmaker"), bm.get("bookmakerName"), bm.get("provider"),
                                bm.get("sportsbook"), bm.get("source"), bm.get("key"), bm.get("code")
                            ] if v] }
                            for bm in (m0.get("bookmakers") or [])[:2]
                        ]
                    }
        print(json.dumps({"sample_event": ev_preview}, indent=2))

    props: List[Dict[str, Any]] = []
    for ev in events:
        props.extend(extract_player_props_from_event(ev, api_key, args.bookmaker, args.debug))
        if len(props) >= args.limit * 2:
            break

    # If empty and user filtered to FanDuel, try again without bookmaker filter (to confirm data exists)
    attempted_bm = bool(args.bookmaker)
    if not props and attempted_bm:
        if args.debug:
            print("[DEBUG] No props found with bookmaker filter; retrying with any bookmaker...")
        for ev in events:
            props.extend(extract_player_props_from_event(ev, api_key, "", args.debug))
            if len(props) >= args.limit * 2:
                break

    # Deduplicate by (game, player, market, line)
    seen = set()
    unique: List[Dict[str, Any]] = []
    for p in props:
        key = (p.get("game"), p.get("player"), p.get("market"), p.get("line"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
        if len(unique) >= args.limit:
            break

    if not unique:
        if attempted_bm and not args.debug:
            print("No player props found for FanDuel with current events. Re-run with --debug or --bookmaker \"\" to diagnose.", file=sys.stderr)
        else:
            print("No player props found for the current events/bookmaker filter.", file=sys.stderr)
        sys.exit(3)

    print(json.dumps(unique[: args.limit], indent=2))


if __name__ == "__main__":
    main()