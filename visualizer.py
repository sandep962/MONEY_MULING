from pyvis.network import Network
import tempfile
import os

# Color scheme for fraud types
FRAUD_COLORS = {
    'legitimate': {'node': '#97C2FC', 'edge': '#CCCCCC'},   # Light blue / gray
    'fan-out':    {'node': '#FF4444', 'edge': '#FF4444'},   # Red
    'fan-in':     {'node': '#FF8800', 'edge': '#FF8800'},   # Orange
    'cycle':      {'node': '#AA00FF', 'edge': '#AA00FF'},   # Purple
    'chain':      {'node': '#FFDD00', 'edge': '#FFDD00'},   # Yellow
}

def create_network_viz(df, height="600px", width="100%", directed=True):
    """
    Creates an interactive Pyvis network graph from a transaction dataframe.
    Color-codes nodes/edges by fraud_type if available.
    Returns the path to the temporary HTML file.
    """
    net = Network(height=height, width=width, directed=directed, notebook=False)
    
    has_fraud_type = 'fraud_type' in df.columns
    has_is_fraud = 'is_fraud' in df.columns
    
    # Simple limit for visualization performance
    if len(df) > 500:
        # Prioritize fraud transactions in the sample
        if has_is_fraud:
            fraud_df = df[df['is_fraud'] == True]
            legit_df = df[df['is_fraud'] == False]
            remaining = max(0, 500 - len(fraud_df))
            if remaining > 0 and len(legit_df) > 0:
                legit_sample = legit_df.sample(min(remaining, len(legit_df)))
                df = pd.concat([fraud_df, legit_sample])
            else:
                df = fraud_df.head(500)
        else:
            df = df.sample(500)
        
    added_nodes = set()
    
    # Pre-calculate node fraud status (a node is fraudulent if ANY of its txns are)
    node_fraud_type = {}
    if has_fraud_type:
        for _, row in df.iterrows():
            ft = row['fraud_type']
            if ft != 'legitimate':
                node_fraud_type[row['sender']] = ft
                node_fraud_type[row['receiver']] = ft
    
    # Calculate node amounts for labels
    sender_amounts = df.groupby('sender')['amount'].sum()
    receiver_amounts = df.groupby('receiver')['amount'].sum()
    total_amounts = sender_amounts.add(receiver_amounts, fill_value=0)
    
    for _, row in df.iterrows():
        fraud_type = row.get('fraud_type', 'legitimate') if has_fraud_type else 'legitimate'
        is_fraud = row.get('is_fraud', False) if has_is_fraud else False
        
        # Source Node
        if row['sender'] not in added_nodes:
            nft = node_fraud_type.get(row['sender'], 'legitimate')
            color = FRAUD_COLORS.get(nft, FRAUD_COLORS['legitimate'])['node']
            label = f"{row['sender']}\n${total_amounts.get(row['sender'], 0):,.2f}"
            title = f"Total Volume: ${total_amounts.get(row['sender'], 0):,.2f}"
            if nft != 'legitimate':
                title += f"\n⚠️ Fraud Type: {nft.upper()}"
            net.add_node(row['sender'], label=label, color=color, title=title, 
                        borderWidth=3 if nft != 'legitimate' else 1)
            added_nodes.add(row['sender'])
            
        # Target Node
        if row['receiver'] not in added_nodes:
            nft = node_fraud_type.get(row['receiver'], 'legitimate')
            color = FRAUD_COLORS.get(nft, FRAUD_COLORS['legitimate'])['node']
            label = f"{row['receiver']}\n${total_amounts.get(row['receiver'], 0):,.2f}"
            title = f"Total Volume: ${total_amounts.get(row['receiver'], 0):,.2f}"
            if nft != 'legitimate':
                title += f"\n⚠️ Fraud Type: {nft.upper()}"
            net.add_node(row['receiver'], label=label, color=color, title=title,
                        borderWidth=3 if nft != 'legitimate' else 1)
            added_nodes.add(row['receiver'])
            
        # Edge
        edge_color = FRAUD_COLORS.get(fraud_type, FRAUD_COLORS['legitimate'])['edge']
        edge_title = f"Amount: ${row['amount']:,.2f}\nDate: {row['timestamp']}"
        if fraud_type != 'legitimate':
            edge_title += f"\n⚠️ {fraud_type.upper()}"
        
        edge_width = max(1, row['amount'] / 1000 + 1)
        if is_fraud:
            edge_width = max(edge_width, 3)  # Make fraud edges thicker
            
        net.add_edge(row['sender'], row['receiver'], color=edge_color, 
                     title=edge_title, width=edge_width)

    # Physics options for better layout
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "iterations": 150 }
      }
    }
    """)
    
    # Create temp file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "fraud_viz.html")
    net.write_html(temp_path)
    
    return temp_path

if __name__ == "__main__":
    from data_manager import generate_transaction_data
    df = generate_transaction_data(n_entities=50, n_transactions=100)
    path = create_network_viz(df)
    print(f"Visualization saved to: {path}")
