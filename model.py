import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import pickle

class LightweightRecommender:
    def __init__(self):
        self.page_descriptions = {
            '1': 'Home Page', 'fontawesome-webfont': 'Font Library',
            'contestproblem': 'Contest Problems', 'login': 'Login Page',
            'profile': 'User Profile', 'details': 'Problem Details',
            'description': 'Problem Description', 'contestsubmission': 'Contest Submissions',
            'archive': 'Problem Archive', 'standings': 'Contest Standings',
            'showcode': 'View Source Code', 'submit': 'Submit Solution',
            'allsubmission': 'All Submissions', 'countdown': 'Contest Timer'
        }
        self.pattern_db = defaultdict(list)
        self.page_similarity = {}
        
    def build_recommendation_model(self, sessions):
        """Build recommendation model using co-occurrence patterns"""
        print("Building recommendation patterns...")
        
        # Build co-occurrence patterns
        page_cooccurrence = defaultdict(Counter)
        
        for session in sessions:
            for i, page in enumerate(session):
                # Pages that come after this page
                if i < len(session) - 1:
                    next_page = session[i + 1]
                    self.pattern_db[page].append(('next', next_page))
                
                # All pages in same session (related)
                for other_page in session:
                    if other_page != page:
                        page_cooccurrence[page][other_page] += 1
        
        # Calculate similarity scores
        all_pages = list(set([page for session in sessions for page in session]))
        
        for page in all_pages:
            total_occurrences = sum(page_cooccurrence[page].values())
            if total_occurrences > 0:
                for related_page, count in page_cooccurrence[page].items():
                    similarity = count / total_occurrences
                    self.pattern_db[page].append(('related', related_page, similarity))
        
        # Sort patterns by frequency
        for page in self.pattern_db:
            # Group by type and page, keep highest similarity
            best_patterns = {}
            for pattern_type, target_page, *similarity in self.pattern_db[page]:
                key = (pattern_type, target_page)
                sim = similarity[0] if similarity else 1.0
                if key not in best_patterns or sim > best_patterns[key]:
                    best_patterns[key] = sim
            
            # Convert back to list and sort
            self.pattern_db[page] = [(pt, tp, sim) for (pt, tp), sim in best_patterns.items()]
            self.pattern_db[page].sort(key=lambda x: x[2], reverse=True)
        
        print(f"Built patterns for {len(self.pattern_db)} pages")
    
    def get_recommendations(self, current_page, top_n=5):
        """Get recommendations for current page"""
        if current_page not in self.pattern_db:
            return self._get_fallback_recommendations(current_page)
        
        recommendations = []
        
        # Get next page recommendations
        next_pages = [(page, sim * 0.8) for typ, page, sim in self.pattern_db[current_page] 
                     if typ == 'next']
        
        # Get related page recommendations  
        related_pages = [(page, sim) for typ, page, sim in self.pattern_db[current_page] 
                        if typ == 'related' and page != current_page]
        
        # Combine and deduplicate
        all_recs = {}
        for page, score in next_pages + related_pages:
            if page not in all_recs or score > all_recs[page]:
                all_recs[page] = score
        
        # Convert to list and sort
        recommendations = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)
        
        # Add contextual fallbacks if needed
        if len(recommendations) < top_n:
            contextual = self._get_contextual_recommendations(current_page)
            for page in contextual:
                if page not in [r[0] for r in recommendations]:
                    recommendations.append((page, 0.3))
        
        return recommendations[:top_n]
    
    def _get_contextual_recommendations(self, current_page):
        """Contextual fallback recommendations"""
        contextual_map = {
            'contestsubmission': ['showcode', 'standings', 'contestproblem'],
            'contestproblem': ['details', 'description', 'submit'],
            'details': ['description', 'submit'],
            'description': ['submit', 'contestsubmission'],
            'submit': ['contestsubmission', 'showcode'],
            'login': ['1', 'profile'],
            'profile': ['contestsubmission', 'standings'],
            '1': ['contestproblem', 'archive', 'login'],
            'archive': ['contestproblem', 'details'],
            'showcode': ['contestsubmission', 'standings']
        }
        return contextual_map.get(current_page, [])
    
    def _get_fallback_recommendations(self, current_page):
        """Fallback when no patterns exist"""
        popular_pages = ['1', 'contestproblem', 'login', 'profile', 'archive']
        return [(page, 0.2) for page in popular_pages if page != current_page][:5]
    
    def describe_page(self, page):
        return self.page_descriptions.get(page, f"{page}")

def parse_web_log_data(file_path):
    """Quick data parsing"""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # Simple column mapping
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'ip' in col_lower: column_mapping[col] = 'IP'
        elif 'time' in col_lower: column_mapping[col] = 'Time' 
        elif 'url' in col_lower: column_mapping[col] = 'URL'
        elif 'status' in col_lower: column_mapping[col] = 'Status'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    return df

def extract_page_from_url(url):
    """Quick page extraction"""
    if pd.isna(url): return "unknown"
    
    url_str = str(url)
    if '://' in url_str:
        parsed = urlparse(url_str)
        path = parsed.path
    else:
        path = url_str
    
    if path == '/' or path == '': return 'home'
    
    path = path.split('?')[0]
    page = path.split('/')[-1]
    
    if page == '': return 'home'
    elif '.' in page: return page.split('.')[0]
    return page

def preprocess_web_log_data(df):
    """Fast preprocessing"""
    df = df.dropna()
    df = df[df['Status'].astype(str).str.isnumeric()]
    df['Status'] = df['Status'].astype(int)
    df = df[df['Status'] == 200]
    
    df['page'] = df['URL'].apply(extract_page_from_url)
    
    # Create sessions by IP
    sessions = df.groupby('IP')['page'].apply(list).tolist()
    valid_sessions = [session for session in sessions if len(session) > 1]
    
    print(f"Created {len(valid_sessions)} sessions")
    return valid_sessions

def display_menu(recommender):
    """Display interactive menu"""
    available_pages = list(recommender.pattern_db.keys())
    
    print("\n" + "="*50)
    print("LIGHTWEIGHT RECOMMENDATION SYSTEM")
    print("="*50)
    
    if not available_pages:
        print("No patterns available. Please check your data.")
        return
    
    for i, page in enumerate(available_pages[:15], 1):
        desc = recommender.describe_page(page)
        print(f"  {i:2d}. {page:<20} - {desc}")
    
    print(f"  {len(available_pages)+1:2d}. Exit")
    print("-"*50)

def main():
    """Main function - lightweight and efficient"""
    file_path = r"C:\Users\Niharika\Desktop\ml-mimi\web-log\weblog.csv"
    
    print("🚀 Loading web log data...")
    try:
        df = parse_web_log_data(file_path)
        sessions = preprocess_web_log_data(df)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    if len(sessions) == 0:
        print("No valid sessions found!")
        return
    
    # Build recommender
    recommender = LightweightRecommender()
    recommender.build_recommendation_model(sessions)
    
    print("✅ Recommendation model ready!")
    
    # Interactive loop
    while True:
        display_menu(recommender)
        available_pages = list(recommender.pattern_db.keys())
        
        if not available_pages:
            break
            
        try:
            choice = input("\nEnter page number: ").strip()
            if choice.isdigit():
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(available_pages):
                    current_page = available_pages[choice_num - 1]
                    recommendations = recommender.get_recommendations(current_page)
                    
                    print(f"\n🎯 Recommendations after: {recommender.describe_page(current_page)}")
                    print("-" * 40)
                    
                    for i, (page, confidence) in enumerate(recommendations, 1):
                        desc = recommender.describe_page(page)
                        print(f"{i}. {desc:<25} - {confidence:.1%} confidence")
                        
                elif choice_num == len(available_pages) + 1:
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice!")
            else:
                print("❌ Please enter a number")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()