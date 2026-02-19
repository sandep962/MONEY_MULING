import networkx as nx
import pandas as pd

def build_graph(df):
    """
    Converts a transaction dataframe into a NetworkX directed graph.
    Handles both labeled and unlabeled data.
    """
    G = nx.MultiDiGraph()
    
    has_fraud_col = 'is_fraud' in df.columns
    has_fraud_type = 'fraud_type' in df.columns
    
    # Add nodes and edges
    for _, row in df.iterrows():
        is_fraud = row.get('is_fraud', False) if has_fraud_col else False
        fraud_type = row.get('fraud_type', 'legitimate') if has_fraud_type else 'legitimate'
        
        # Source node
        if not G.has_node(row['sender']):
            G.add_node(row['sender'], is_fraud=False, fraud_type='legitimate')
        
        # Target node
        if not G.has_node(row['receiver']):
            G.add_node(row['receiver'], is_fraud=False, fraud_type='legitimate')
            
        # Update fraud status if either node is part of a fraudulent transaction
        if is_fraud:
            G.nodes[row['sender']]['is_fraud'] = True
            G.nodes[row['receiver']]['is_fraud'] = True
            if fraud_type != 'legitimate':
                G.nodes[row['sender']]['fraud_type'] = fraud_type
                G.nodes[row['receiver']]['fraud_type'] = fraud_type
            
        # Add edge
        G.add_edge(
            row['sender'], 
            row['receiver'], 
            amount=row['amount'], 
            timestamp=row['timestamp'],
            is_fraud=is_fraud,
            fraud_type=fraud_type
        )
        
    return G

def calculate_metrics(G):
    """
    Calculates graph-based metrics for fraud detection.
    """
    # Convert MultiDiGraph to DiGraph for certain algorithms (simple weighted edges)
    simple_G = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if simple_G.has_edge(u, v):
            simple_G[u][v]['weight'] += data['amount']
        else:
            simple_G.add_edge(u, v, weight=data['amount'])
            
    # PageRank (influence/centrality)
    pagerank = nx.pagerank(simple_G, weight='weight')
    
    # Out-degree/In-degree ratios
    out_degree = dict(G.out_degree())
    in_degree = dict(G.in_degree())
    
    # Identify potential suspicious nodes (high PageRank or unusual ratios)
    metrics = []
    for node in G.nodes():
        metrics.append({
            'node': node,
            'pagerank': round(pagerank.get(node, 0), 6),
            'out_degree': out_degree.get(node, 0),
            'in_degree': in_degree.get(node, 0),
            'is_fraud': G.nodes[node].get('is_fraud', False),
            'fraud_type': G.nodes[node].get('fraud_type', 'legitimate')
        })
        
    return pd.DataFrame(metrics)

if __name__ == "__main__":
    from data_manager import generate_transaction_data
    df = generate_transaction_data(n_entities=100, n_transactions=500)
    G = build_graph(df)
    metrics_df = calculate_metrics(G)
    print(metrics_df.sort_values(by='pagerank', ascending=False).head())
