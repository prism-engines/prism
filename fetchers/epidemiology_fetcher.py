#!/usr/bin/env python3
"""
Delphi Epidemiology Fetcher

Standalone fetcher for CMU Delphi Epidata API.
Does NOT import from prism - this is a workspace script.

No API key required for most endpoints.

Data sources:
- CDC FluView (Influenza-like Illness)
- FluSurv-NET (Hospitalizations)
- Wikipedia access logs

Usage:
    from fetchers.delphi_fetcher import fetch

    config = {
        "signals": ["ILI_NATIONAL", "ILI_HHS1", "FLU_HOSP_NATIONAL"],
    }
    observations = fetch(config)
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


SOURCE = "delphi"
BASE_URL = "https://api.delphi.cmu.edu/epidata"


# =============================================================================
# INDICATOR DEFINITIONS
# =============================================================================

INDICATORS = {
    # CDC FluView - Influenza surveillance (national + 10 HHS regions)
    "ILI_NATIONAL": {"endpoint": "fluview", "params": {"regions": "nat"}},
    "ILI_HHS1": {"endpoint": "fluview", "params": {"regions": "hhs1"}},
    "ILI_HHS2": {"endpoint": "fluview", "params": {"regions": "hhs2"}},
    "ILI_HHS3": {"endpoint": "fluview", "params": {"regions": "hhs3"}},
    "ILI_HHS4": {"endpoint": "fluview", "params": {"regions": "hhs4"}},
    "ILI_HHS5": {"endpoint": "fluview", "params": {"regions": "hhs5"}},
    "ILI_HHS6": {"endpoint": "fluview", "params": {"regions": "hhs6"}},
    "ILI_HHS7": {"endpoint": "fluview", "params": {"regions": "hhs7"}},
    "ILI_HHS8": {"endpoint": "fluview", "params": {"regions": "hhs8"}},
    "ILI_HHS9": {"endpoint": "fluview", "params": {"regions": "hhs9"}},
    "ILI_HHS10": {"endpoint": "fluview", "params": {"regions": "hhs10"}},

    # Hospitalizations
    "FLU_HOSP_NATIONAL": {"endpoint": "flusurv", "params": {"locations": "network_all"}},

    # Wikipedia (proxy for public awareness)
    "WIKI_FLU": {"endpoint": "wiki", "params": {"articles": "influenza"}},
    "WIKI_COVID": {"endpoint": "wiki", "params": {"articles": "covid-19"}},
}


# =============================================================================
# HELPERS
# =============================================================================

def epiweek_to_date(epiweek: int) -> pd.Timestamp:
    """Convert CDC epiweek (YYYYWW) to date."""
    year = epiweek // 100
    week = epiweek % 100

    jan1 = datetime(year, 1, 1)
    days_to_sunday = (6 - jan1.weekday()) % 7
    first_sunday = jan1 + timedelta(days=days_to_sunday)
    target_date = first_sunday + timedelta(weeks=week - 1)

    return pd.Timestamp(target_date)


# =============================================================================
# ENDPOINT FETCHERS
# =============================================================================

def fetch_fluview(signal_id: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch from FluView endpoint (ILI rates)."""
    url = f"{BASE_URL}/fluview/"

    current_year = datetime.now().year
    start_week = (current_year - 20) * 100 + 1
    end_week = current_year * 100 + 52

    request_params = {
        "regions": params.get("regions", "nat"),
        "epiweeks": f"{start_week}-{end_week}",
    }

    response = requests.get(url, params=request_params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if data.get("result") != 1:
        return []

    observations = []
    for row in data.get("epidata", []):
        epiweek = row.get("epiweek")
        ili = row.get("wili") or row.get("ili")

        if epiweek and ili is not None:
            observations.append({
                "signal_id": signal_id,
                "observed_at": epiweek_to_date(epiweek),
                "value": float(ili),
                "source": SOURCE,
            })

    return observations


def fetch_flusurv(signal_id: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch from FluSurv-NET endpoint (hospitalization rates)."""
    url = f"{BASE_URL}/flusurv/"

    current_year = datetime.now().year
    start_week = (current_year - 10) * 100 + 1
    end_week = current_year * 100 + 52

    request_params = {
        "locations": params.get("locations", "network_all"),
        "epiweeks": f"{start_week}-{end_week}",
    }

    response = requests.get(url, params=request_params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if data.get("result") != 1:
        return []

    observations = []
    for row in data.get("epidata", []):
        epiweek = row.get("epiweek")
        rate = row.get("rate_overall") or row.get("rate")

        if epiweek and rate is not None:
            observations.append({
                "signal_id": signal_id,
                "observed_at": epiweek_to_date(epiweek),
                "value": float(rate),
                "source": SOURCE,
            })

    return observations


def fetch_wiki(signal_id: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch Wikipedia access data."""
    url = f"{BASE_URL}/wiki/"

    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)

    request_params = {
        "articles": params.get("articles", "influenza"),
        "dates": f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}",
    }

    response = requests.get(url, params=request_params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if data.get("result") != 1:
        return []

    observations = []
    for row in data.get("epidata", []):
        date_str = str(row.get("date"))
        count = row.get("count") or row.get("value")

        if date_str and count is not None:
            try:
                date = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
                observations.append({
                    "signal_id": signal_id,
                    "observed_at": date,
                    "value": float(count),
                    "source": SOURCE,
                })
            except ValueError:
                continue

    return observations


# =============================================================================
# MAIN FETCH FUNCTION
# =============================================================================

def fetch(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetch observations for signals specified in config.

    Args:
        config: Dict with keys:
            - signals: list of signal IDs (required)

    Returns:
        List of observation dicts with keys:
            signal_id, observed_at, value, source
    """
    signals = config.get("signals", [])
    if not signals:
        raise ValueError("Config must contain 'signals' list")

    all_observations = []

    for signal_id in signals:
        if signal_id not in INDICATORS:
            print(f"  {signal_id}: FAILED - Unknown signal")
            continue

        ind_config = INDICATORS[signal_id]
        endpoint = ind_config["endpoint"]
        params = ind_config.get("params", {})

        try:
            if endpoint == "fluview":
                obs = fetch_fluview(signal_id, params)
            elif endpoint == "flusurv":
                obs = fetch_flusurv(signal_id, params)
            elif endpoint == "wiki":
                obs = fetch_wiki(signal_id, params)
            else:
                print(f"  {signal_id}: FAILED - Unknown endpoint: {endpoint}")
                continue

            all_observations.extend(obs)
            print(f"  {signal_id}: {len(obs)} observations")

        except Exception as e:
            print(f"  {signal_id}: FAILED - {e}")

    return all_observations


if __name__ == "__main__":
    config = {
        "signals": ["ILI_NATIONAL", "ILI_HHS1"],
    }
    results = fetch(config)
    print(f"\nTotal: {len(results)} observations")
