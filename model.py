import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re
from urllib.parse import urlparse, parse_qs
import time

class TreeNode:
    """Node class for FP-tree"""
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None  # for header table linkage

class FPTree:
    """FP-tree implementation with enhanced visualization"""
    
    def __init__(self, min_support=0.01):
        self.min_support = min_support
        self.header_table = {}
        self.root = TreeNode(None, 1, None)
        self.frequent_itemsets = []
    
    def _build_header_table(self, transactions):
        """Build initial header table with item frequencies"""
        item_freq = Counter()
        for transaction in transactions:
            for item in transaction:
                item_freq[item] += 1
        
        # Filter items based on minimum support
        min_count = self.min_support * len(transactions)
        self.header_table = {item: [count, None] 
                           for item, count in item_freq.items() 
                           if count >= min_count}
        
        # Sort items by frequency descending
        self.frequent_items = sorted(self.header_table.keys(), 
                                   key=lambda x: self.header_table[x][0], 
                                   reverse=True)
        return self.frequent_items
    
    def _update_tree(self, transaction, count=1):
        """Update the FP-tree with a transaction"""
        node = self.root
        
        # Sort transaction items by frequency
        sorted_items = [item for item in self.frequent_items 
                       if item in transaction]
        sorted_items.sort(key=lambda x: self.header_table[x][0], reverse=True)
        
        for item in sorted_items:
            if item in node.children:
                child = node.children[item]
                child.count += count
            else:
                child = TreeNode(item, count, node)
                node.children[item] = child
                
                # Update header table linkage
                if self.header_table[item][1] is None:
                    self.header_table[item][1] = child
                else:
                    current = self.header_table[item][1]
                    while current.next is not None:
                        current = current.next
                    current.next = child
            
            node = child
    
    def build_tree(self, transactions):
        """Build the complete FP-tree from transactions"""
        # Build header table and get frequent items
        frequent_items = self._build_header_table(transactions)
        
        # Build tree
        for transaction in transactions:
            self._update_tree(transaction)
        
        print(f"FP-tree built with {len(frequent_items)} frequent items")
        return self
    
    def _get_conditional_pattern_base(self, item):
        """Get conditional pattern base for an item"""
        pattern_base = []
        node = self.header_table[item][1]
        
        while node is not None:
            prefix_path = []
            parent = node.parent
            
            # Traverse up to root
            while parent.item is not None:
                prefix_path.append(parent.item)
                parent = parent.parent
            
            if prefix_path:
                pattern_base.append((prefix_path, node.count))
            
            node = node.next
        
        return pattern_base
    
    def _mine_tree(self, suffix, min_count):
        """Recursively mine the FP-tree"""
        # Process items in ascending order of frequency
        items = [item for item in self.header_table.keys()]
        items.sort(key=lambda x: self.header_table[x][0])
        
        for item in items:
            new_suffix = [item] + suffix
            support = self.header_table[item][0]
            
            if support >= min_count:
                self.frequent_itemsets.append((new_suffix, support))
                
                # Build conditional pattern base
                pattern_base = self._get_conditional_pattern_base(item)
                
                if pattern_base:
                    # Build conditional FP-tree
                    conditional_transactions = []
                    for pattern, count in pattern_base:
                        conditional_transactions.extend([pattern] * count)
                    
                    conditional_tree = FPTree(self.min_support)
                    conditional_tree.build_tree(conditional_transactions)
                    
                    # Mine conditional tree
                    if conditional_tree.header_table:
                        conditional_tree._mine_tree(new_suffix, min_count)
    
    def mine_frequent_itemsets(self):
        """Mine all frequent itemsets from the FP-tree"""
        min_count = self.min_support * self._get_transaction_count()
        self.frequent_itemsets = []
        self._mine_tree([], min_count)
        return self.frequent_itemsets
    
    def _get_transaction_count(self):
        """Estimate transaction count from root"""
        count = 0
        for child in self.root.children.values():
            count += child.count
        return count
class UserRecommendationSystem:
    def __init__(self, frequent_itemsets, total_transactions):
        self.frequent_itemsets = frequent_itemsets
        self.total_transactions = total_transactions
        self.page_descriptions = {
            '1': 'Home Page',
            'fontawesome-webfont': 'Font Library',
            'contestproblem': 'Contest Problems',
            'login': 'Login Page',
            'profile': 'User Profile',
            'details': 'Problem Details',
            'description': 'Problem Description',
            'contestsubmission': 'Contest Submissions',
            'archive': 'Problem Archive',
            'standings': 'Contest Standings',
            'showcode': 'View Source Code',
            'submit': 'Submit Solution'
        }
        
        # Build enhanced pattern database
        self.pattern_database = self._build_enhanced_pattern_database()
        self.global_popular_pages = self._get_global_popular_pages()
    
    def _build_enhanced_pattern_database(self):
        """Build enhanced pattern database with better pattern detection"""
        pattern_db = defaultdict(list)
        
        for itemset, support in self.frequent_itemsets:
            if len(itemset) >= 1:  # Include single-page patterns for context
                support_pct = (support / self.total_transactions) * 100
                
                # For multi-page patterns, extract sequences
                if len(itemset) > 1:
                    for i in range(len(itemset)):
                        current_page = itemset[i]
                        # What comes after this page
                        if i < len(itemset) - 1:
                            next_page = itemset[i + 1]
                            pattern_db[current_page].append(('next', next_page, support_pct))
                        # What comes before this page
                        if i > 0:
                            prev_page = itemset[i - 1]
                            pattern_db[current_page].append(('prev', prev_page, support_pct))
                        # All related pages in this pattern
                        for j, related_page in enumerate(itemset):
                            if j != i:
                                pattern_db[current_page].append(('related', related_page, support_pct))
        
        # Sort and clean the pattern database
        for page in pattern_db:
            # Remove duplicates and sort by support
            unique_patterns = {}
            for pattern_type, target_page, support in pattern_db[page]:
                key = (pattern_type, target_page)
                if key not in unique_patterns or support > unique_patterns[key]:
                    unique_patterns[key] = support
            
            # Convert back to list and sort
            pattern_db[page] = [(pt, tp, sup) for (pt, tp), sup in unique_patterns.items()]
            pattern_db[page].sort(key=lambda x: x[2], reverse=True)
        
        return pattern_db
    
    def _get_global_popular_pages(self):
        """Get globally popular pages as fallback recommendations"""
        page_support = defaultdict(float)
        
        for itemset, support in self.frequent_itemsets:
            support_pct = (support / self.total_transactions) * 100
            for page in itemset:
                page_support[page] = max(page_support[page], support_pct)
        
        return sorted(page_support.items(), key=lambda x: x[1], reverse=True)
    
    def get_next_page_recommendations(self, current_page, top_n=5):
        """Get enhanced next page recommendations"""
        recommendations = []
        
        # Get direct next page patterns
        if current_page in self.pattern_database:
            next_pages = [(page, sup) for typ, page, sup in self.pattern_database[current_page] 
                         if typ == 'next']
            recommendations.extend(next_pages)
        
        # If no direct patterns, use related pages as potential next steps
        if not recommendations and current_page in self.pattern_database:
            related_pages = [(page, sup) for typ, page, sup in self.pattern_database[current_page] 
                           if typ == 'related']
            recommendations.extend(related_pages[:top_n])
        
        # Fallback to globally popular pages
        if not recommendations:
            recommendations = [(page, sup) for page, sup in self.global_popular_pages 
                             if page != current_page][:top_n]
        
        # Remove duplicates and return top N
        unique_recs = {}
        for page, support in recommendations:
            if page not in unique_recs or support > unique_recs[page]:
                unique_recs[page] = support
        
        return sorted(unique_recs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_related_pages(self, current_page, top_n=5):
        """Get enhanced related page recommendations"""
        related = []
        
        if current_page in self.pattern_database:
            # Get all related pages from patterns
            for typ, page, support in self.pattern_database[current_page]:
                if typ in ['related', 'prev', 'next'] and page != current_page:
                    related.append((page, support))
        
        # Remove duplicates and sort
        unique_related = {}
        for page, support in related:
            if page not in unique_related or support > unique_related[page]:
                unique_related[page] = support
        
        results = sorted(unique_related.items(), key=lambda x: x[1], reverse=True)
        
        # Fallback if no related pages found
        if not results:
            results = [(page, sup) for page, sup in self.global_popular_pages 
                      if page != current_page][:top_n]
        
        return results[:top_n]
    
    def get_contextual_recommendations(self, current_page, top_n=3):
        """Get contextual recommendations based on page type"""
        contextual_map = {
            'archive': ['contestproblem', 'details', 'description', 'standings'],
            'contestproblem': ['details', 'description', 'submit', 'archive'],
            'details': ['description', 'submit', 'contestproblem'],
            'description': ['submit', 'contestsubmission', 'showcode'],
            'submit': ['contestsubmission', 'showcode', 'standings'],
            'login': ['1', 'profile', 'contestproblem'],
            'profile': ['contestsubmission', 'standings', 'archive'],
            '1': ['contestproblem', 'archive', 'standings', 'login']
        }
        
        if current_page in contextual_map:
            contextual_pages = contextual_map[current_page]
            # Enhance with actual support data if available
            enhanced_recs = []
            for page in contextual_pages:
                support = self._get_support_for_page(page)
                enhanced_recs.append((page, support))
            return sorted(enhanced_recs, key=lambda x: x[1], reverse=True)[:top_n]
        
        return []
    
    def _get_support_for_page(self, page):
        """Get support percentage for a specific page"""
        for itemset, support in self.frequent_itemsets:
            if page in itemset:
                return (support / self.total_transactions) * 100
        return 0.0
    
    def get_popular_flows(self, min_support=3.0):
        """Get popular user navigation flows"""
        popular_flows = []
        for itemset, support in self.frequent_itemsets:
            if len(itemset) > 1:
                support_pct = (support / self.total_transactions) * 100
                if support_pct >= min_support:
                    flow_description = " → ".join([self.page_descriptions.get(page, page) for page in itemset])
                    popular_flows.append((flow_description, support_pct))
        
        return sorted(popular_flows, key=lambda x: x[1], reverse=True)
    
    def describe_page(self, page):
        """Get description for a page"""
        return self.page_descriptions.get(page, f"Unknown Page: {page}")
    
    def get_all_available_pages(self):
        """Get list of all available pages from frequent itemsets"""
        all_pages = set()
        for itemset, _ in self.frequent_itemsets:
            for page in itemset:
                all_pages.add(page)
        return sorted(list(all_pages))

def display_recommendations(recommendation_system, current_page):
    """Display enhanced recommendations for the current page"""
    print(f"\n🎯 RECOMMENDATIONS for: {recommendation_system.describe_page(current_page)}")
    print("="*50)
    
    # Get next page recommendations
    next_recs = recommendation_system.get_next_page_recommendations(current_page)
    if next_recs:
        print("\n📈 NEXT PAGE SUGGESTIONS (Based on user patterns):")
        for i, (page, support) in enumerate(next_recs, 1):
            description = recommendation_system.describe_page(page)
            print(f"  {i}. {description:<25} - {support:.1f}% of users")
    else:
        print("\n📈 No specific next page patterns found for this page.")
    
    # Get related pages
    related_recs = recommendation_system.get_related_pages(current_page)
    if related_recs:
        print("\n🔗 RELATED PAGES (Frequently visited together):")
        for i, (page, support) in enumerate(related_recs, 1):
            description = recommendation_system.describe_page(page)
            print(f"  {i}. {description:<25} - {support:.1f}% co-occurrence")
    
    # Get contextual recommendations
    contextual_recs = recommendation_system.get_contextual_recommendations(current_page)
    if contextual_recs:
        print("\n💡 CONTEXTUAL SUGGESTIONS (Based on page type):")
        for i, (page, support) in enumerate(contextual_recs, 1):
            description = recommendation_system.describe_page(page)
            support_text = f" - {support:.1f}% support" if support > 0 else ""
            print(f"  {i}. {description:<25}{support_text}")
    
    # Show user message if no recommendations
    if not next_recs and not related_recs and not contextual_recs:
        print("\n🤔 No specific patterns found for this page.")
        print("   Try exploring popular user flows to see common navigation paths!")
    
    print("\n" + "-"*50)

def display_popular_flows(recommendation_system):
    """Display popular user navigation flows"""
    print(f"\n🏆 POPULAR USER NAVIGATION FLOWS")
    print("="*60)
    
    popular_flows = recommendation_system.get_popular_flows(min_support=2.0)  # Lower threshold
    
    if popular_flows:
        print("Common user navigation patterns:")
        for i, (flow, support) in enumerate(popular_flows, 1):
            print(f"{i:2d}. {flow}")
            print(f"    📊 Used by {support:.1f}% of users")
            print()
    else:
        print("No popular flows found. Try lowering the support threshold.")
        print("\nMost frequent individual pages:")
        for i, (page, support) in enumerate(recommendation_system.global_popular_pages[:10], 1):
            desc = recommendation_system.describe_page(page)
            print(f"  {i}. {desc:<25} - {support:.1f}%")

# Update the menu display function
def display_recommendation_menu(recommendation_system):
    """Display the recommendation menu"""
    print("\n" + "="*60)
    print("        CODING CONTEST PLATFORM RECOMMENDATION SYSTEM")
    print("="*60)
    
    available_pages = recommendation_system.get_all_available_pages()
    print("Available Pages:")
    for i, page in enumerate(available_pages, 1):
        description = recommendation_system.describe_page(page)
        print(f"  {i:2d}. {page:<20} - {description}")
    
    print(f"  {len(available_pages)+1:2d}. View Popular User Flows")
    print(f"  {len(available_pages)+2:2d}. Exit")
    print("-"*60)

def parse_web_log_data(file_path):
    """Parse the web log data from CSV"""
    print("Loading and parsing web log data...")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Display basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for column names and clean them
    df.columns = df.columns.str.strip()
    
    # Rename columns based on common web log formats
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'ip' in col_lower:
            column_mapping[col] = 'IP'
        elif 'time' in col_lower or 'date' in col_lower:
            column_mapping[col] = 'Time'
        elif 'url' in col_lower:
            column_mapping[col] = 'URL'
        elif 'status' in col_lower:
            column_mapping[col] = 'Status'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"Renamed columns: {column_mapping}")
    
    print(f"Final columns: {df.columns.tolist()}")
    return df

def extract_page_from_url(url):
    """Extract page name from URL"""
    if pd.isna(url):
        return "unknown"
    
    url_str = str(url)
    
    # Extract path from URL
    if '://' in url_str:
        parsed = urlparse(url_str)
        path = parsed.path
    else:
        path = url_str
    
    # Extract page name
    if path == '/' or path == '':
        return 'home'
    
    # Remove query parameters
    path = path.split('?')[0]
    
    # Get the last part of the path as page name
    page = path.split('/')[-1]
    
    if page == '':
        page = 'home'
    elif '.' in page:
        page = page.split('.')[0]  # Remove file extensions
    
    return page

def preprocess_web_log_data(df):
    """Preprocess web log data for FP-tree analysis"""
    print("\nPreprocessing data...")
    
    # Clean the data
    df = df.dropna()
    df = df[df['Status'].astype(str).str.isnumeric()]
    df['Status'] = df['Status'].astype(int)
    
    # Filter successful requests (status code 200)
    df = df[df['Status'] == 200]
    print(f"After filtering successful requests: {len(df)} rows")
    
    # Extract page names from URLs
    df['page'] = df['URL'].apply(extract_page_from_url)
    
    # Parse timestamps
    def parse_timestamp(time_str):
        try:
            # Handle different timestamp formats
            time_str = str(time_str).strip()
            if time_str.startswith('['):
                time_str = time_str[1:]
            if ':' in time_str and '/' in time_str:
                # Format: [29/Nov/2017:06:58:55
                return pd.to_datetime(time_str, format='%d/%b/%Y:%H:%M:%S', errors='coerce')
            else:
                return pd.to_datetime(time_str, errors='coerce')
        except:
            return pd.NaT
    
    df['timestamp'] = df['Time'].apply(parse_timestamp)
    df = df.dropna(subset=['timestamp'])
    
    # Sort by IP and timestamp
    df = df.sort_values(['IP', 'timestamp'])
    
    # Create sessions based on IP and 30-minute timeout
    df['time_diff'] = df.groupby('IP')['timestamp'].diff().dt.total_seconds().fillna(0)
    df['new_session'] = (df['time_diff'] > 1800).astype(int)  # 30 minutes
    df['session_id'] = df.groupby('IP')['new_session'].cumsum()
    
    # Create composite session identifier
    df['full_session_id'] = df['IP'].astype(str) + '_' + df['session_id'].astype(str)
    
    # Create transactions: unique pages visited in each session (in order)
    sessions = df.groupby('full_session_id').agg({
        'page': list,
        'timestamp': 'min',
        'IP': 'first'
    }).reset_index()
    
    # Remove single-page sessions and filter out very long sessions
    sessions = sessions[sessions['page'].apply(len) > 1]
    sessions = sessions[sessions['page'].apply(len) <= 50]  # Remove outliers
    
    print(f"Created {len(sessions)} valid sessions")
    print(f"Average session length: {sessions['page'].apply(len).mean():.2f} pages")
    
    # Display most common pages
    page_counts = Counter()
    for pages in sessions['page']:
        page_counts.update(pages)
    
    print(f"\nTop 10 most frequent pages:")
    for page, count in page_counts.most_common(10):
        print(f"  {page}: {count} occurrences")
    
    return sessions['page'].tolist(), df, sessions

def analyze_frequent_patterns(transactions, min_support=0.01):
    """Complete FP-tree analysis pipeline"""
    
    print("=" * 60)
    print("FP-Tree Frequent Pattern Analysis")
    print("=" * 60)
    
    if len(transactions) == 0:
        print("No transactions to analyze!")
        return None, []
    
    # Build FP-tree
    start_time = time.time()
    fp_tree = FPTree(min_support=min_support)
    fp_tree.build_tree(transactions)
    build_time = time.time() - start_time
    
    print(f"Tree building time: {build_time:.2f} seconds")
    
    # Mine frequent itemsets
    start_time = time.time()
    frequent_itemsets = fp_tree.mine_frequent_itemsets()
    mine_time = time.time() - start_time
    
    print(f"Mining time: {mine_time:.2f} seconds")
    print(f"Total frequent itemsets found: {len(frequent_itemsets)}")
    
    return fp_tree, frequent_itemsets

def display_recommendation_menu(recommendation_system):
    """Display the recommendation menu"""
    print("\n" + "="*60)
    print("        CODING CONTEST PLATFORM RECOMMENDATION SYSTEM")
    print("="*60)
    
    available_pages = recommendation_system.get_all_available_pages()
    print("Available Pages:")
    for i, page in enumerate(available_pages, 1):
        description = recommendation_system.describe_page(page)
        print(f"  {i:2d}. {page:<20} - {description}")
    
    print(f"  {len(available_pages)+1:2d}. View Popular User Flows")
    print(f"  {len(available_pages)+2:2d}. Exit")
    print("-"*60)

def get_user_choice(recommendation_system):
    """Get user's current page choice"""
    available_pages = recommendation_system.get_all_available_pages()
    
    while True:
        try:
            choice = input("\nEnter your current page number: ").strip()
            
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_pages):
                    return available_pages[choice_num - 1]
                elif choice_num == len(available_pages) + 1:
                    return 'popular_flows'
                elif choice_num == len(available_pages) + 2:
                    return 'exit'
            
            print(f"Invalid choice! Please enter a number between 1-{len(available_pages)+2}")
        except KeyboardInterrupt:
            return 'exit'
        except Exception as e:
            print(f"Error: {e}. Please try again.")

def display_recommendations(recommendation_system, current_page):
    """Display recommendations for the current page"""
    print(f"\n🎯 RECOMMENDATIONS for: {recommendation_system.describe_page(current_page)}")
    print("="*50)
    
    # Get next page recommendations
    next_recs = recommendation_system.get_next_page_recommendations(current_page)
    if next_recs:
        print("\n📈 NEXT PAGE SUGGESTIONS (Based on user patterns):")
        for i, (page, support) in enumerate(next_recs, 1):
            description = recommendation_system.describe_page(page)
            print(f"  {i}. {description:<25} - {support:.1f}% of users")
    else:
        print("\n📈 No specific next page patterns found for this page.")
    
    # Get related pages
    related_recs = recommendation_system.get_related_pages(current_page)
    if related_recs:
        print("\n🔗 RELATED PAGES (Frequently visited together):")
        for i, (page, support) in enumerate(related_recs, 1):
            description = recommendation_system.describe_page(page)
            print(f"  {i}. {description:<25} - {support:.1f}% co-occurrence")
    
    print("\n" + "-"*50)

def display_popular_flows(recommendation_system):
    """Display popular user navigation flows"""
    print(f"\n🏆 POPULAR USER NAVIGATION FLOWS")
    print("="*60)
    
    popular_flows = recommendation_system.get_popular_flows(min_support=3.0)
    if popular_flows:
        for i, (flow, support) in enumerate(popular_flows, 1):
            print(f"{i:2d}. {flow}")
            print(f"    📊 Used by {support:.1f}% of users")
            print()
    else:
        print("No popular flows found with the current support threshold.")

def run_recommendation_engine(frequent_itemsets, total_transactions):
    """Run the interactive recommendation engine"""
    recommendation_system = UserRecommendationSystem(frequent_itemsets, total_transactions)
    
    print("\n" + "⭐" * 20)
    print("RECOMMENDATION ENGINE STARTED!")
    print("⭐" * 20)
    
    while True:
        display_recommendation_menu(recommendation_system)
        choice = get_user_choice(recommendation_system)
        
        if choice == 'exit':
            print("\nThank you for using the Recommendation System! Goodbye! 👋")
            break
        elif choice == 'popular_flows':
            display_popular_flows(recommendation_system)
        else:
            display_recommendations(recommendation_system, choice)
        
        input("\nPress Enter to continue...")

def main():
    """Main function to run complete FP-tree analysis and recommendations"""
    
    # Load and parse the web log data
    file_path = r"C:\Users\Niharika\Desktop\ml-mimi\web-log\weblog.csv"
    
    try:
        df = parse_web_log_data(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found!")
        return
    
    # Preprocess data
    transactions, processed_df, sessions_df = preprocess_web_log_data(df)
    
    if len(transactions) == 0:
        print("No valid transactions found after preprocessing!")
        return
    
    # Run FP-tree analysis
    min_support = 0.02  # Adjust based on data size
    
    print(f"\n{'='*70}")
    print(f"Analysis with Minimum Support: {min_support*100}%")
    print(f"{'='*70}")
    
    fp_tree, frequent_itemsets = analyze_frequent_patterns(transactions, min_support)
    
    if fp_tree and frequent_itemsets:
        # Save results
        results_df = pd.DataFrame([
            {'itemset': itemset, 'support': support, 'support_pct': (support/len(transactions))*100} 
            for itemset, support in frequent_itemsets
        ])
        results_df = results_df.sort_values('support', ascending=False)
        
        filename = f'frequent_patterns_support_{min_support}.csv'
        results_df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        
        # Start interactive recommendation engine
        print(f"\nStarting Recommendation Engine with {len(frequent_itemsets)} patterns...")
        run_recommendation_engine(frequent_itemsets, len(transactions))

if __name__ == "__main__":
    main()