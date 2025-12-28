import pandas as pd

PATH = "reports/generation_eval_v4_rows.csv"

df = pd.read_csv(PATH)   # plus besoin de engine python ni on_bad_lines
df.columns = df.columns.str.strip()


print("Detected columns:", df.columns.tolist())

ft = df[df["method"] == "finetuned_v4"].copy()

# Check 1
neg = ft[ft["rating"] == 3].copy()
n_total = len(neg)
n_pred_pos = (neg["pred_label"] == 1).sum()
share = n_pred_pos / n_total if n_total > 0 else float("nan")

print("\n=== Check 1: rating=3 but predicted positive ===")
print(f"Total rating=3 rows: {n_total}")
print(f"pred_label=1 count:  {n_pred_pos}")
print(f"Share:               {share:.3f}")

# Check 2
neg["pred_proba_recommended"] = pd.to_numeric(neg["pred_proba_recommended"], errors="coerce")
susp = neg[neg["pred_proba_recommended"] > 0.8].copy()
susp = susp.sort_values("pred_proba_recommended", ascending=False).head(5)

print("\n=== Check 2: top 5 rating=3 with pred_proba_recommended > 0.8 ===")
for _, row in susp.iterrows():
    print("\n---")
    print(f"Title: {row.get('title','')}")
    print(f"pred_proba_recommended: {row['pred_proba_recommended']:.3f}")
    print(row.get("generated_text", ""))
