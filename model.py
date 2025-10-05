import pandas as pd
import re
from mlxtend.frequent_patterns import fpgrowth, association_rules

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
file_path = r"C:\Users\Niharika\Desktop\ml-mimi\web-log\weblog.csv"  # Replace with your downloaded CSV path
df = pd.read_csv(file_path)
df.columns = [c.strip() for c in df.columns]

# -------------------------------
# Step 2: Preprocess
# -------------------------------
# Keep only rows with successful requests
if 'Status' in df.columns:
    df = df[df['Status'].astype(str).str.isdigit()]
    df['Status'] = df['Status'].astype(int)
    df = df[df['Status'] == 200]

# Extract page name from URL
def extract_page(url):
    match = re.search(r'/([\w\-]+)\.(php|html|asp|aspx|js|css)', str(url))
    return match.group(1) if match else None

df['Page'] = df['URL'].apply(extract_page)
df = df[df['Page'].notna()]

# -------------------------------
# Step 3: Create sessions by IP
# -------------------------------
transactions = df.groupby('IP')['Page'].apply(list)

# -------------------------------
# Step 4: One-hot encode transactions
# -------------------------------
all_pages = df['Page'].value_counts()
top_pages = all_pages[all_pages >= 5].index.tolist()  # Keep pages visited at least 5 times

encoded_data = []
for session in transactions:
    row = {page: (page in session) for page in top_pages}
    encoded_data.append(row)

encoded_df = pd.DataFrame(encoded_data)

# -------------------------------
# Step 5: FP-Growth
# -------------------------------
frequent_itemsets = fpgrowth(encoded_df, min_support=0.1, use_colnames=True, max_len=3)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

# -------------------------------
# Step 6: Association Rules
# -------------------------------
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules.sort_values(by=['confidence', 'lift'], ascending=False)

# -------------------------------
# Step 7: Recommendation function
# -------------------------------
def recommend_next_page(current_page, top_n=5):
    related_rules = rules[rules['antecedents'].apply(lambda x: current_page in x)]
    if related_rules.empty:
        return [("No strong association found", 0)]

    related_rules = related_rules[['consequents', 'confidence', 'lift']].head(top_n)

    recommendations = []
    for _, row in related_rules.iterrows():
        for page in row['consequents']:
            recommendations.append((page, row['confidence']))

    # Remove duplicates
    unique_recs = {}
    for page, conf in recommendations:
        if page not in unique_recs or conf > unique_recs[page]:
            unique_recs[page] = conf

    results = sorted(unique_recs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return results


# After preprocessing
print("All unique pages in the dataset:")
all_pages = df['Page'].unique()
for i, page in enumerate(all_pages, 1):
    print(f"{i}. {page}")
# -------------------------------
# Step 8: Test recommendation
# -------------------------------
current_page = "announcement"  # Example page
print(f"Recommendations after visiting '{current_page}':")
for page, conf in recommend_next_page(current_page):
    print(f"- {page:20} | Confidence: {conf*100:.1f}%")
