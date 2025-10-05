import pandas as pd
import re
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import fpgrowth, association_rules

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
file_path = r"C:\Users\Niharika\Desktop\ml-mimi\web-log\weblog.csv"
df = pd.read_csv(file_path)
df.columns = [c.strip() for c in df.columns]

# Fix typo
if 'Staus' in df.columns:
    df.rename(columns={'Staus': 'Status'}, inplace=True)

# -------------------------------
# Step 2: Preprocess
# -------------------------------
if 'Status' in df.columns:
    df = df[df['Status'].astype(str).str.isdigit()]
    df['Status'] = df['Status'].astype(int)
    df = df[df['Status'] == 200]

def extract_page(url):
    match = re.search(r'/([\w\-]+)', str(url))
    return match.group(1) if match else None

df['Page'] = df['URL'].apply(extract_page)
df = df[df['Page'].notna()]

# -------------------------------
# Step 3: Create realistic sessions
# -------------------------------
# Each IP’s visit sequence is broken into smaller sessions (to add variation)
sessions = []
for ip, group in df.groupby('IP'):
    pages = list(group['Page'])
    # break into sub-sessions of size up to 10 pages
    for i in range(0, len(pages), 10):
        sub_session = list(set(pages[i:i+10]))
        if len(sub_session) > 1:
            sessions.append(sub_session)

print(f"🧩 Total Sessions Formed: {len(sessions)}")

# -------------------------------
# Step 4: One-hot Encode
# -------------------------------
page_counts = df['Page'].value_counts()
filtered_pages = page_counts[(page_counts > 3) & (page_counts < 400)].index.tolist()

encoded_data = []
for session in sessions:
    row = {page: (page in session) for page in filtered_pages}
    encoded_data.append(row)
encoded_df = pd.DataFrame(encoded_data).fillna(False)

# -------------------------------
# Step 5: FP-Growth Mining
# -------------------------------
print("\n⏳ Mining Frequent Itemsets...")
frequent_itemsets = fpgrowth(encoded_df, min_support=0.05, use_colnames=True, max_len=3)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
print(f"✅ Frequent Itemsets Found: {len(frequent_itemsets)}")

print("\nFrequent Itemsets Preview:")
print(frequent_itemsets.head(10).to_string(index=False))

# -------------------------------
# Step 6: Association Rules
# -------------------------------
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
rules = rules[(rules['lift'] > 1) & (rules['confidence'] < 1)]
rules = rules.sort_values(by=['confidence', 'lift'], ascending=False)

print("\nGenerated Rules Table:")
print(rules[['antecedents','consequents','support','confidence','lift']].head(10).to_string(index=False))

# -------------------------------
# Step 7: Visualizations
# -------------------------------
top_rules = rules.head(10)
plt.figure(figsize=(8,5))
plt.barh(
    [f"{list(a)[0]} → {list(c)[0]}" for a, c in zip(top_rules['antecedents'], top_rules['consequents'])],
    top_rules['confidence'],
)
plt.xlabel("Confidence")
plt.title("Top 10 Association Rules (by Confidence)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Network Graph
G = nx.DiGraph()
for _, row in top_rules.iterrows():
    for a in row['antecedents']:
        for c in row['consequents']:
            G.add_edge(a, c, weight=row['confidence'])

plt.figure(figsize=(7,5))
pos = nx.spring_layout(G, k=0.7)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1500,
        font_size=9, width=[w['weight']*3 for u,v,w in G.edges(data=True)],
        edge_color='gray', arrows=True)
plt.title("Association Rule Network")
plt.show()

# -------------------------------
# Step 8: Recommendation Function
# -------------------------------
def recommend_next_page(current_page, top_n=3):
    related_rules = rules[rules['antecedents'].apply(lambda x: current_page in x)]
    if related_rules.empty:
        return [("⚠️ No strong recommendations found", 0)]
    related_rules = related_rules[['consequents','confidence']].head(top_n)
    recs = []
    for _, row in related_rules.iterrows():
        for p in row['consequents']:
            recs.append((p, row['confidence']))
    return recs

# -------------------------------
# Step 9: Output Demo
# -------------------------------
print("\n#️⃣ Dataset Summary")
print(f"Unique Pages: {len(filtered_pages)}")

print("\n#️⃣ Top 10 Rules (by Confidence)")
for i, (a, c, conf, lift) in enumerate(zip(
        top_rules['antecedents'], top_rules['consequents'],
        top_rules['confidence'], top_rules['lift']), 1):
    print(f"{i}. {set(a)} → {set(c)} | Confidence: {conf:.2f}, Lift: {lift:.2f}")

print("\n#️⃣ Recommendation Examples")
for page in ["home", "announcement", "update","allsubmission"]:
    print(f"\nUser visits: {page}")
    for rec, conf in recommend_next_page(page):
        print(f"✅ Recommended Next: {rec} | Confidence: {conf*100:.1f}%")

rules[['antecedents','consequents','support','confidence','lift']].to_csv("association_rules.csv", index=False)
print("\n💾 association_rules.csv saved successfully!")
