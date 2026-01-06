import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import pandas as pd
import requests
from tqdm import tqdm
from src.config import TRAIN_APP_IDS, VALIDATION_APP_IDS, FILES

REVIEWS_PER_GAME = 2000
REQUEST_SLEEP = 1.0

def fetch_reviews(app_ids, output_path):
    print(f"--- Collecte des reviews pour {len(app_ids)} jeux ---")
    rows = []
    for app_id in tqdm(app_ids, desc="Downloading Reviews"):
        cursor = "*"
        collected = 0
        while collected < REVIEWS_PER_GAME:
            try:
                url = f"https://store.steampowered.com/appreviews/{app_id}"
                params = {"json": 1, "filter": "recent", "language": "english", "num_per_page": 100, "cursor": cursor}
                resp = requests.get(url, params=params, timeout=10)
                data = resp.json()
                
                reviews = data.get("reviews", [])
                if not reviews: break
                cursor = data.get("cursor", cursor)

                for r in reviews:
                    rows.append({
                        "app_id": app_id,
                        "recommended": r.get("voted_up"),
                        "review_text": r.get("review", ""),
                        "votes_up": r.get("votes_up")
                    })
                    collected += 1
                    if collected >= REVIEWS_PER_GAME: break
                time.sleep(REQUEST_SLEEP)
            except Exception as e:
                print(f"Error on app {app_id}: {e}")
                break
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Sauvegardé : {output_path} ({len(df)} lignes)")

def fetch_titles(app_ids, output_path):
    print(f"--- Récupération des titres pour {len(app_ids)} jeux ---")
    data = []
    with requests.Session() as s:
        for app_id in tqdm(app_ids, desc="Fetching Titles"):
            try:
                resp = s.get("https://store.steampowered.com/api/appdetails", params={"appids": app_id}, timeout=10)
                if resp.status_code == 200:
                    json_data = resp.json()
                    if json_data[str(app_id)]["success"]:
                        data.append({"app_id": app_id, "title": json_data[str(app_id)]["data"]["name"]})
            except Exception:
                pass
            time.sleep(0.5)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Sauvegardé : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "validation"], help="Quel dataset construire ?")
    args = parser.parse_args()

    target_ids = TRAIN_APP_IDS if args.mode == "train" else VALIDATION_APP_IDS
    paths = FILES[args.mode]

    fetch_reviews(target_ids, paths["raw_reviews"])
    fetch_titles(target_ids, paths["titles"])