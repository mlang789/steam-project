import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# A small starter list of well-known Steam app IDs (you can change them anytime)
APP_IDS = [
    570,      # Dota 2
    730,      # Counter-Strike 2
    440,      # Team Fortress 2
    1091500,  # Cyberpunk 2077
    1245620,  # Elden Ring
]

REVIEWS_PER_GAME = 2000     # how many reviews to collect per game
REQUEST_SLEEP_SEC = 1.0     # sleep between requests to avoid spamming
LANGUAGE = "english"        # keep it consistent for generation and modeling

Path("data/raw").mkdir(parents=True, exist_ok=True)

rows = []

for app_id in tqdm(APP_IDS, desc="Collecting reviews"):
    cursor = "*"
    collected = 0

    while collected < REVIEWS_PER_GAME:
        url = f"https://store.steampowered.com/appreviews/{app_id}"
        params = {
            "json": 1,
            "filter": "recent",
            "language": LANGUAGE,
            "num_per_page": 100,
            "cursor": cursor,
        }

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        cursor = data.get("cursor", cursor)
        reviews = data.get("reviews", [])

        # Safety: sometimes you may get no reviews back (end of pagination, etc.)
        if not reviews:
            break

        for review in reviews:
            author = review.get("author", {}) or {}

            rows.append({
                "app_id": app_id,
                "language": review.get("language"),
                "recommended": review.get("voted_up"),
                "review_text": review.get("review", ""),
                "timestamp_created": review.get("timestamp_created"),
                "playtime_at_review": author.get("playtime_at_review", None),

                # Extra useful metadata (optional but nice)
                "steam_purchase": review.get("steam_purchase", None),
                "received_for_free": review.get("received_for_free", None),
                "written_during_early_access": review.get("written_during_early_access", None),
                "votes_up": review.get("votes_up", None),
                "votes_funny": review.get("votes_funny", None),
                "weighted_vote_score": review.get("weighted_vote_score", None),
            })

            collected += 1
            if collected >= REVIEWS_PER_GAME:
                break

        time.sleep(REQUEST_SLEEP_SEC)

df = pd.DataFrame(rows)
output_path = "data/raw/reviews_raw.csv"
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"Saved {output_path} with shape: {df.shape}")
