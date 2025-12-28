import pandas as pd

PATH = "reports/generation_eval_v4_rows.csv"

df = pd.read_csv(PATH)
df.columns = df.columns.str.strip()

ft = df[df["method"] == "finetuned_v4"].copy()
ft["rating"] = pd.to_numeric(ft["rating"], errors="coerce")
ft["pred_proba_recommended"] = pd.to_numeric(ft["pred_proba_recommended"], errors="coerce")
ft["pred_label"] = pd.to_numeric(ft["pred_label"], errors="coerce")

neg = ft[ft["rating"] == 3].copy()

print("Total finetuned_v4 rating=3:", len(neg))
print("Predicted positive (pred_label=1):", int((neg["pred_label"] == 1).sum()))

# 1) Show 10 random disagreements (rating=3 but predicted positive)
dis = neg[neg["pred_label"] == 1].sample(min(10, len(neg)), random_state=0)

print("\n=== 10 examples: rating=3 but predicted positive ===")
for _, r in dis.iterrows():
    print("\n---")
    print("Title:", r.get("title", ""))
    print("pred_proba:", float(r["pred_proba_recommended"]))
    print(r["generated_text"])

# 2) Rule-based quick check: does the text explicitly say "do not recommend"?
neg["has_not_recommend"] = neg["generated_text"].str.lower().str.contains("do not recommend|don't recommend|not recommend", regex=True)
print("\nShare of rating=3 that explicitly says NOT recommend:", neg["has_not_recommend"].mean())

# Among disagreements, how many explicitly say NOT recommend?
dis2 = neg[neg["pred_label"] == 1].copy()
dis2["has_not_recommend"] = dis2["generated_text"].str.lower().str.contains("do not recommend|don't recommend|not recommend", regex=True)
print("Among predicted-positive rating=3, share saying NOT recommend:", dis2["has_not_recommend"].mean())
