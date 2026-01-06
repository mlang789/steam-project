import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.config import EXTERNAL_OUTPUTS_DIR, GENAI_INPUTS_DIR

INPUT_DIR = EXTERNAL_OUTPUTS_DIR
OUT_PATH = GENAI_INPUTS_DIR / "prompt_batch_filled.csv"

FILES_MAP = {
    "naive": "generated_reviews_naive.csv",
    "engineered": "generated_reviews_engineered.csv",
    "finetuned": "generated_reviews_finetuned.csv"
}

def main():
    print(f"--- Fusion des fichiers depuis : {INPUT_DIR} ---")
    
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Le dossier {INPUT_DIR} est introuvable.")

    dfs = []
    for method, filename in FILES_MAP.items():
        file_path = INPUT_DIR / filename
        if file_path.exists():
            print(f"✅ Chargement de '{method}' : {filename}")
            try:
                df = pd.read_csv(file_path, encoding="utf-8", engine="python")
                df["method"] = method
                if "generated_text" in df.columns:
                    dfs.append(df)
                else:
                    print(f"   ⚠️ Colonne manquante dans {filename}")
            except Exception as e:
                print(f"   ❌ Erreur : {e}")
        else:
            print(f"   ❌ Manquant : {filename}")

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        full_df["generated_text"] = full_df["generated_text"].fillna("").astype(str)
        
        cols_wanted = ["app_id", "title", "rating", "method", "generated_text"]
        cols_present = [c for c in cols_wanted if c in full_df.columns]
        full_df = full_df[cols_present]
        
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(OUT_PATH, index=False, encoding="utf-8")
        print(f"\n🎉 Succès ! {len(full_df)} reviews fusionnées vers {OUT_PATH}")
    else:
        print("\n❌ Aucun fichier fusionné.")

if __name__ == "__main__":
    main()