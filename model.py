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

    def print_tree_detailed(self, node=None, level=0, prefix="", is_last=True):
        """Print detailed FP-tree structure with better visualization"""
        if node is None:
            node = self.root
            print("🎯 DETAILED FP-TREE STRUCTURE")
            print("=" * 80)
            print("Root (count: 1)")
        
        # Create connector symbols
        if level > 0:
            connector = "└── " if is_last else "├── "
            print(prefix + connector + f"{node.item}({node.count})")
            new_prefix = prefix + ("    " if is_last else "│   ")
        else:
            new_prefix = prefix
        
        # Print children
        children_items = list(node.children.items())
        for i, (child_item, child_node) in enumerate(children_items):
            is_last_child = (i == len(children_items) - 1)
            self.print_tree_detailed(child_node, level + 1, new_prefix, is_last_child)

    def print_tree_compact(self):
        """Print compact FP-tree representation"""
        print("\n📊 COMPACT FP-TREE REPRESENTATION")
        print("=" * 60)
        
        def print_node_compact(node, level=0):
            indent = "  " * level
            if node.item is None:
                print(f"{indent}ROOT")
            else:
                print(f"{indent}├─ {node.item} (count: {node.count})")
            
            for child in node.children.values():
                print_node_compact(child, level + 1)
        
        print_node_compact(self.root)

    def print_header_table(self):
        """Print the header table structure"""
        print("\n📋 HEADER TABLE")
        print("=" * 50)
        print(f"{'Item':<20} {'Frequency':<12} {'Node Chain'}")
        print("-" * 50)
        
        for item, (freq, first_node) in sorted(self.header_table.items(), 
                                             key=lambda x: x[1][0], reverse=True):
            node_chain = []
            current = first_node
            while current:
                node_chain.append(f"{current.item}({current.count})")
                current = current.next
            
            chain_str = " → ".join(node_chain) if node_chain else "None"
            print(f"{item:<20} {freq:<12} {chain_str}")

    def print_tree_statistics(self):
        """Print statistics about the FP-tree"""
        print("\n📈 FP-TREE STATISTICS")
        print("=" * 40)
        
        total_nodes = self._count_nodes(self.root)
        max_depth = self._get_max_depth(self.root)
        avg_branching = self._get_avg_branching(self.root)
        
        print(f"Total nodes: {total_nodes}")
        print(f"Maximum depth: {max_depth}")
        print(f"Average branching factor: {avg_branching:.2f}")
        print(f"Frequent items in header: {len(self.header_table)}")
        print(f"Root children: {len(self.root.children)}")

    def _count_nodes(self, node):
        """Count total nodes in the tree"""
        if node is None:
            return 0
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def _get_max_depth(self, node):
        """Get maximum depth of the tree"""
        if not node.children:
            return 0
        return 1 + max(self._get_max_depth(child) for child in node.children.values())

    def _get_avg_branching(self, node):
        """Calculate average branching factor"""
        if not node.children:
            return 0
        
        total_children = 0
        total_nodes_with_children = 0
        
        def traverse(n):
            nonlocal total_children, total_nodes_with_children
            if n.children:
                total_children += len(n.children)
                total_nodes_with_children += 1
                for child in n.children.values():
                    traverse(child)
        
        traverse(node)
        return total_children / total_nodes_with_children if total_nodes_with_children > 0 else 0

    def print_complete_tree_analysis(self):
        """Print complete tree analysis with all visualizations"""
        print("\n" + "🌳" * 30)
        print("COMPLETE FP-TREE ANALYSIS")
        print("🌳" * 30)
        
        self.print_tree_statistics()
        self.print_header_table()
        print("\n")
        self.print_tree_detailed()
        print("\n")
        self.print_tree_compact()

def parse_web_log_data(file_path):
    """Parse the web log data from CSV"""
    print("Loading and parsing web log data...")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Display basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    
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
    
    print(f"\nFinal columns: {df.columns.tolist()}")
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
    
    # Print complete tree analysis
    fp_tree.print_complete_tree_analysis()
    
    # Mine frequent itemsets
    start_time = time.time()
    frequent_itemsets = fp_tree.mine_frequent_itemsets()
    mine_time = time.time() - start_time
    
    print(f"Mining time: {mine_time:.2f} seconds")
    print(f"Total frequent itemsets found: {len(frequent_itemsets)}")
    
    return fp_tree, frequent_itemsets

def main():
    """Main function to run complete FP-tree analysis"""
    
    # Load and parse the web log data
    file_path = r"C:\Users\Niharika\Desktop\ml-mimi\weblog.csv"  # Update path if needed
    
    try:
        df = parse_web_log_data(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found. Creating sample data for demonstration...")
        # Create sample data for demonstration
        sample_data = {
            'IP': ['10.128.2.1', '10.128.2.1', '10.128.2.1', '10.131.2.1', '10.130.2.1'],
            'Time': ['[29/Nov/2017:06:58:55', '[29/Nov/2017:06:59:02', '[29/Nov/2017:06:59:03', 
                    '[29/Nov/2017:06:59:04', '[29/Nov/2017:06:59:06'],
            'URL': ['GET /login.php HTTP/1.1', 'POST /process.php HTTP/1.1', 'GET /home.php HTTP/1.1',
                   'GET /js/vendor/moment.min.js HTTP/1.1', 'GET /bootstrap-3.3.7/js/bootstrap.js HTTP/1.1'],
            'Status': [200, 302, 200, 200, 200]
        }
        df = pd.DataFrame(sample_data)
        print("Using sample data for demonstration")
    
    # Preprocess data
    transactions, processed_df, sessions_df = preprocess_web_log_data(df)
    
    if len(transactions) == 0:
        print("No valid transactions found after preprocessing!")
        return
    
    # Run FP-tree analysis
    min_support = 0.03  # Adjust based on data size
    
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

if __name__ == "__main__":
    main()