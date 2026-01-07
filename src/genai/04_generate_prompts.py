import sys
from pathlib import Path

# ajout du chemin racine pour trouver src.config
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.config import FILES, GENAI_INPUTS_DIR

DATA_PATH = FILES["train"]["final"]
OUT_PATH = GENAI_INPUTS_DIR / "prompt_batch.csv"

# nombre de prompts à générer par jeu et par type de note
N_PER_GAME_PER_RATING = 1000

# notes utilisées dans le dataset
POS_RATING = 9
NEG_RATING = 3

# descriptions manuelles associées aux identifiants steam
GAME_DESCRIPTIONS_BY_APP_ID = {
    570: (
        "Dota 2 is a free-to-play 5v5 MOBA where two teams fight to destroy the enemy Ancient. "
        "Matches revolve around laning, farming, objectives, teamfights, and a huge roster of heroes with deep mechanics."
    ),
    730: (
        "Counter-Strike 2 is a competitive 5v5 tactical FPS focused on precise gunplay, teamwork, and economy management. "
        "Rounds are usually bomb defusal, where small decisions and coordination can decide the game."
    ),
    440: (
        "Team Fortress 2 is a class-based multiplayer shooter with 9 distinct classes and objective-driven modes like Payload and Control Points. "
        "It has a stylized tone, chaotic fights, and a strong focus on class synergy and funny emergent moments."
    ),
    1091500: (
        "Cyberpunk 2077 is an open-world action RPG set in Night City. "
        "You play in first-person, mix narrative quests with combat and stealth, and build your character through gear, skills, and cyberware."
    ),
    1245620: (
        "Elden Ring is an open-world action RPG by FromSoftware with challenging combat, large-scale exploration, and tough bosses. "
        "It emphasizes learning enemy patterns, building your character, and discovering secrets across a dark fantasy world."
    ),
}

# correspondance de secours via le titre si l'identifiant est manquant
GAME_DESCRIPTIONS_BY_TITLE_KEYWORD = {
    "dota 2": GAME_DESCRIPTIONS_BY_APP_ID[570],
    "counter-strike 2": GAME_DESCRIPTIONS_BY_APP_ID[730],
    "cs2": GAME_DESCRIPTIONS_BY_APP_ID[730],
    "team fortress 2": GAME_DESCRIPTIONS_BY_APP_ID[440],
    "tf2": GAME_DESCRIPTIONS_BY_APP_ID[440],
    "cyberpunk 2077": GAME_DESCRIPTIONS_BY_APP_ID[1091500],
    "elden ring": GAME_DESCRIPTIONS_BY_APP_ID[1245620],
}

def get_game_description(app_id: int, title: str) -> str:
    # 1. priorité à la correspondance par identifiant
    if app_id in GAME_DESCRIPTIONS_BY_APP_ID:
        return GAME_DESCRIPTIONS_BY_APP_ID[app_id]

    # 2. recherche par mot-clé dans le titre
    title_l = (title or "").strip().lower()
    for k, desc in GAME_DESCRIPTIONS_BY_TITLE_KEYWORD.items():
        if k in title_l:
            return desc

    # 3. description générique par défaut
    return (
        "This is a video game on Steam. Write the review based on what a regular player would realistically experience."
    )

def make_naive_prompt(title: str, rating: int) -> str:
    return f'Write a Steam user review for the game "{title}" with rating {rating}/10.'

def make_engineered_prompt(title: str, rating: int, game_description: str) -> str:
    # structure : 2 points positifs et 1 négatif (ou l'inverse selon la note)
    if rating >= 7:
        point_guideline = (
            "Naturally include 2 things you genuinely liked and 1 thing you disliked (worked into the text, not as a list)."
        )
    else:
        point_guideline = (
            "Naturally include 2 things you genuinely disliked and 1 thing you liked (worked into the text, not as a list)."
        )

    return "\n".join(
        [
            "You are a real Steam user writing a review after actually playing the game.",
            "Write as human as possible: include personal experience, small details, and a natural voice.",
            "",
            f'Game: "{title}"',
            f"Target rating: {rating}/10",
            "",
            "Game description (use this to ground your review):",
            game_description,
            "",
            "Guidelines:",
            point_guideline,
            "- Mention at least one concrete personal detail (e.g., playtime estimate, playing solo vs with friends, a memorable moment, performance/controls feel, difficulty spike).",
            "- Sound casual and honest (not marketing). It's okay to be a bit imperfect or nuanced.",
            "- No spoilers (don’t reveal major story beats or endings).",
            "- Output only the review text (no title, no bullets).",
        ]
    )

def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = {"app_id", "title", "rating"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")

    # filtrage pour ne garder que les notes cibles
    df = df[df["rating"].isin([NEG_RATING, POS_RATING])].copy()

    rows = []

    # échantillonnage par groupe de jeu et de note
    for (app_id, title, rating), group in df.groupby(
        ["app_id", "title", "rating"], dropna=False
    ):
        sample_n = min(N_PER_GAME_PER_RATING, len(group))
        if sample_n == 0:
            continue

        sampled = group.sample(n=sample_n, random_state=42)

        # récupération de la description unique pour ce groupe
        safe_app_id = int(app_id) if pd.notna(app_id) else -1
        safe_title = str(title)
        game_desc = get_game_description(safe_app_id, safe_title)

        for _, _row in sampled.iterrows():
            # génération du prompt naïf
            rows.append(
                {
                    "app_id": safe_app_id,
                    "title": safe_title,
                    "rating": int(rating),
                    "method": "naive",
                    "prompt": make_naive_prompt(safe_title, int(rating)),
                    "generated_text": "",
                }
            )

            # génération du prompt travaillé
            rows.append(
                {
                    "app_id": safe_app_id,
                    "title": safe_title,
                    "rating": int(rating),
                    "method": "engineered",
                    "prompt": make_engineered_prompt(safe_title, int(rating), game_desc),
                    "generated_text": "",
                }
            )

    out_df = pd.DataFrame(rows)

    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {OUT_PATH} with shape: {out_df.shape}")
    print(out_df.head(6).to_string(index=False))

if __name__ == "__main__":
    main()