# src/enrich_games.py
"""
Fetch Steam game titles from app IDs found in data/raw/reviews_raw.csv,
then save a mapping file to data/processed/games.csv.

Output CSV columns:
- app_id
- title
"""

import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from tqdm import tqdm


RAW_REVIEWS_PATH = Path("data/raw/reviews_raw.csv")
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "games.csv"

REQUEST_SLEEP_SEC = 0.8
TIMEOUT_SEC = 30
MAX_RETRIES = 3


def fetch_title_for_appid(app_id: int, session: requests.Session) -> Optional[str]:
    """
    Call Steam Store 'appdetails' API to get the game title (data.name).
    Returns None if not found or request fails.
    """
    url = "https://store.steampowered.com/api/appdetails"
    params = {"appids": str(app_id)}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=TIMEOUT_SEC)
            resp.raise_for_status()
            payload = resp.json()

            # Expected shape: { "<appid>": { "success": bool, "data": {...} } }
            app_key = str(app_id)
            if app_key not in payload:
                return None

            entry = payload[app_key]
            if not entry.get("success", False):
                return None

            data = entry.get("data", {}) or {}
            title = data.get("name", None)

            if isinstance(title, str) and title.strip():
                return title.strip()

            return None

        except (requests.RequestException, ValueError):
            # retry after a short delay
            if attempt < MAX_RETRIES:
                time.sleep(0.8 * attempt)
            else:
                return None

    return None


def main() -> None:
    if not RAW_REVIEWS_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {RAW_REVIEWS_PATH}. Run your collection script first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_REVIEWS_PATH, usecols=["app_id"])
    app_ids = sorted(df["app_id"].dropna().astype(int).unique().tolist())

    if not app_ids:
        raise ValueError("No app_id found in the raw reviews file.")

    id_to_title: Dict[int, Optional[str]] = {}

    with requests.Session() as session:
        for app_id in tqdm(app_ids, desc="Fetching game titles"):
            title = fetch_title_for_appid(app_id, session)
            id_to_title[app_id] = title
            time.sleep(REQUEST_SLEEP_SEC)

    out_df = pd.DataFrame(
        [{"app_id": app_id, "title": title} for app_id, title in id_to_title.items()]
    )

    # Basic sanity check: warn if some titles are missing
    missing = out_df["title"].isna().sum()
    if missing > 0:
        print(f"Warning: {missing}/{len(out_df)} titles could not be fetched (title is empty).")

    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {OUT_PATH} with shape: {out_df.shape}")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
