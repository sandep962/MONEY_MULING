"""
Fraud Detection Engine
Detects 4 types of graph-based fraud patterns from unlabeled transaction data:
1. Fan-out: One sender → many receivers in a short time window
2. Fan-in: Many senders → one receiver in a short time window
3. Cycle: Circular money flow (A→B→C→A)
4. Chain: Layered transfers with decreasing amounts (A→S1→S2→B)
"""

import pandas as pd
import networkx as nx
from collections import defaultdict
import json


def detect_all_fraud(df, time_window_days=3, fanout_threshold=10, fanin_threshold=10):
    """
    Runs all fraud detection algorithms on a transaction dataframe.
    Returns the dataframe with 'is_fraud' and 'fraud_type' columns populated.
    """
    df = df.copy()
    df['is_fraud'] = False
    df['fraud_type'] = 'legitimate'

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Run each detector
    df = _detect_fanout(df, time_window_days, fanout_threshold)
    df = _detect_fanin(df, time_window_days, fanin_threshold)
    df = _detect_cycles(df)
    df = _detect_chains(df)

    return df


def _detect_fanout(df, time_window_days=3, threshold=10):
    """
    Detect fan-out fraud: one sender sending to many unique receivers
    within a short time window.
    """
    senders = df['sender'].unique()

    for sender in senders:
        sender_txns = df[df['sender'] == sender].copy()
        if len(sender_txns) < threshold:
            continue

        sender_txns = sender_txns.sort_values('timestamp')
        timestamps = sender_txns['timestamp'].values

        # Sliding window approach
        for i in range(len(sender_txns)):
            start_time = timestamps[i]
            end_time = start_time + pd.Timedelta(days=time_window_days)

            window_mask = (sender_txns['timestamp'] >= start_time) & (sender_txns['timestamp'] <= end_time)
            window_txns = sender_txns[window_mask]

            unique_receivers = window_txns['receiver'].nunique()

            if unique_receivers >= threshold:
                # Flag all transactions in this window for this sender
                flagged_indices = window_txns.index
                df.loc[flagged_indices, 'is_fraud'] = True
                df.loc[flagged_indices, 'fraud_type'] = df.loc[flagged_indices, 'fraud_type'].apply(
                    lambda x: 'fan-out' if x == 'legitimate' else x
                )
                break  # One detection per sender is enough

    return df


def _detect_fanin(df, time_window_days=3, threshold=10):
    """
    Detect fan-in fraud: many unique senders sending to one receiver
    within a short time window.
    """
    receivers = df['receiver'].unique()

    for receiver in receivers:
        recv_txns = df[df['receiver'] == receiver].copy()
        if len(recv_txns) < threshold:
            continue

        recv_txns = recv_txns.sort_values('timestamp')
        timestamps = recv_txns['timestamp'].values

        for i in range(len(recv_txns)):
            start_time = timestamps[i]
            end_time = start_time + pd.Timedelta(days=time_window_days)

            window_mask = (recv_txns['timestamp'] >= start_time) & (recv_txns['timestamp'] <= end_time)
            window_txns = recv_txns[window_mask]

            unique_senders = window_txns['sender'].nunique()

            if unique_senders >= threshold:
                flagged_indices = window_txns.index
                df.loc[flagged_indices, 'is_fraud'] = True
                df.loc[flagged_indices, 'fraud_type'] = df.loc[flagged_indices, 'fraud_type'].apply(
                    lambda x: 'fan-in' if x == 'legitimate' else x
                )
                break

    return df


def _detect_cycles(df):
    """
    Detect cycle fraud: circular money flow (e.g., A→B→C→A).
    Uses NetworkX cycle detection on the transaction graph.
    """
    # Build a simple DiGraph for cycle detection
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['sender'], row['receiver'])

    # Find simple cycles of length >= 2
    try:
        cycles = list(nx.simple_cycles(G))
    except Exception:
        return df

    # Filter to meaningful cycles (length >= 3)
    fraud_cycles = [c for c in cycles if len(c) >= 3]

    # Flag transactions that are part of any cycle
    cycle_nodes = set()
    for cycle in fraud_cycles:
        cycle_nodes.update(cycle)

    if cycle_nodes:
        # Flag edges that form part of a cycle
        for cycle in fraud_cycles:
            for i in range(len(cycle)):
                sender = cycle[i]
                receiver = cycle[(i + 1) % len(cycle)]
                mask = (df['sender'] == sender) & (df['receiver'] == receiver)
                df.loc[mask, 'is_fraud'] = True
                df.loc[mask, 'fraud_type'] = df.loc[mask, 'fraud_type'].apply(
                    lambda x: 'cycle' if x == 'legitimate' else x
                )

    return df


def _detect_chains(df, min_chain_length=3, amount_decay_ratio=0.98):
    """
    Detect chain/layered fraud: money moves through intermediaries
    with amounts decreasing along the path (A→S1→S2→B).
    Skips nodes already flagged as other fraud types to avoid duplicates.
    """
    # Build a weighted DiGraph
    G = nx.DiGraph()
    edge_data = defaultdict(list)

    for idx, row in df.iterrows():
        key = (row['sender'], row['receiver'])
        edge_data[key].append({'idx': idx, 'amount': row['amount'], 'timestamp': row['timestamp']})
        if not G.has_edge(row['sender'], row['receiver']):
            G.add_edge(row['sender'], row['receiver'], amount=row['amount'])

    # Find nodes that are both senders and receivers (intermediaries)
    senders = set(df['sender'].unique())
    receivers = set(df['receiver'].unique())
    intermediaries = senders & receivers

    # For each potential chain start (nodes that are only senders, or have few incoming)
    potential_starts = senders - receivers  # Pure sources
    if not potential_starts:
        # Fallback: use nodes with more outgoing than incoming
        for node in senders:
            out_d = G.out_degree(node) if G.has_node(node) else 0
            in_d = G.in_degree(node) if G.has_node(node) else 0
            if out_d > in_d:
                potential_starts.add(node)

    flagged_chain_indices = set()

    for start in potential_starts:
        # DFS to find chains with decreasing amounts
        _find_chains_dfs(G, edge_data, start, [start], None, min_chain_length, amount_decay_ratio, flagged_chain_indices)

    # Apply flags
    for idx in flagged_chain_indices:
        if df.loc[idx, 'fraud_type'] == 'legitimate':
            df.loc[idx, 'is_fraud'] = True
            df.loc[idx, 'fraud_type'] = 'chain'

    return df


def _find_chains_dfs(G, edge_data, current, path, prev_amount, min_length, decay_ratio, flagged):
    """DFS helper to find chain fraud patterns."""
    if len(path) > 10:  # Safety limit
        return

    neighbors = list(G.successors(current))
    for neighbor in neighbors:
        if neighbor in path:  # Avoid cycles (handled separately)
            continue

        key = (current, neighbor)
        if key not in edge_data:
            continue

        for txn in edge_data[key]:
            amount = txn['amount']

            # Check if amount decreases (layered laundering pattern)
            if prev_amount is None or amount <= prev_amount * (1 / decay_ratio):
                new_path = path + [neighbor]

                if len(new_path) >= min_length + 1:  # +1 because path includes nodes, not edges
                    # Flag all edges in this chain
                    for i in range(len(new_path) - 1):
                        edge_key = (new_path[i], new_path[i + 1])
                        if edge_key in edge_data:
                            for t in edge_data[edge_key]:
                                flagged.add(t['idx'])

                # Continue searching deeper
                _find_chains_dfs(G, edge_data, neighbor, new_path, amount, min_length, decay_ratio, flagged)


def get_fraud_summary(df):
    """
    Returns a summary of detected fraud patterns.
    """
    fraud_df = df[df['is_fraud']].copy()
    if fraud_df.empty:
        return pd.DataFrame(columns=['fraud_type', 'count', 'total_amount'])

    summary = fraud_df.groupby('fraud_type').agg(
        count=('amount', 'size'),
        total_amount=('amount', 'sum')
    ).reset_index()
    summary['total_amount'] = summary['total_amount'].round(2)

    return summary


def generate_json_report(df, processing_time_seconds):
    """
    Generates a structured JSON report of detected fraud, including
    suspicious accounts, fraud rings, and a summary.
    """
    total_accounts = pd.concat([df['sender'], df['receiver']]).nunique()
    
    fraud_df = df[df['is_fraud']]
    fraud_G = nx.DiGraph()
    
    account_patterns = defaultdict(set)
    account_fraud_amount = defaultdict(float)
    
    for _, row in fraud_df.iterrows():
        u, v = row['sender'], row['receiver']
        f_type = row['fraud_type']
        amt = row['amount']
        
        fraud_G.add_edge(u, v, fraud_type=f_type)
        
        account_patterns[u].add(f_type)
        account_patterns[v].add(f_type)
        account_fraud_amount[u] += amt
        account_fraud_amount[v] += amt
        
    rings = []
    account_to_ring = {}
    
    if fraud_G.number_of_nodes() > 0:
        # Identify weakly connected components as 'rings'
        components = list(nx.weakly_connected_components(fraud_G))
        for i, comp in enumerate(components):
            ring_id = f"RING_{i+1:03d}"
            
            comp_patterns = []
            comp_amount = 0
            for node in comp:
                # Ensure node is a native Python string or int for JSON serialization
                node_native = str(node) if not isinstance(node, (int, float, str)) else node
                account_to_ring[node_native] = ring_id
                
                comp_patterns.extend(list(account_patterns[node]))
                comp_amount += account_fraud_amount[node]
                
            if comp_patterns:
                pattern_type = max(set(comp_patterns), key=comp_patterns.count)
            else:
                pattern_type = "unknown"
                
            risk_score = min(100.0, 50.0 + len(comp) * 2 + comp_amount / 1000)
            
            rings.append({
                "ring_id": ring_id,
                # JSON requires native types
                "member_accounts": [str(n) if not isinstance(n, (int, float, str)) else n for n in comp],
                "pattern_type": pattern_type,
                "risk_score": round(risk_score, 1)
            })
            
    suspicious_accounts = []
    
    for acc in account_patterns.keys():
        score = min(100.0, 60.0 + len(account_patterns[acc]) * 10 + account_fraud_amount[acc] / 500)
        # Ensure native types
        acc_native = str(acc) if not isinstance(acc, (int, float, str)) else acc
        suspicious_accounts.append({
            "account_id": acc_native,
            "suspicion_score": round(score, 1),
            "detected_patterns": list(account_patterns[acc]),
            "ring_id": account_to_ring.get(acc_native, "")
        })
        
    suspicious_accounts.sort(key=lambda x: x["suspicion_score"], reverse=True)
    
    summary = {
        "total_accounts_analyzed": int(total_accounts),
        "suspicious_accounts_flagged": len(suspicious_accounts),
        "fraud_rings_detected": len(rings),
        "processing_time_seconds": round(processing_time_seconds, 2)
    }
    
    report = {
        "suspicious_accounts": suspicious_accounts,
        "fraud_rings": rings,
        "summary": summary
    }
    
    return json.dumps(report, indent=2)


if __name__ == "__main__":
    from data_manager import process_uploaded_csv

    print("=== Testing Fraud Detection on transactionss.csv ===\n")
    df = process_uploaded_csv(r"c:\Users\Jyoti Shetti\OneDrive\Desktop\rift\transactionss.csv")
    result = detect_all_fraud(df)

    print(f"Total transactions: {len(result)}")
    print(f"Flagged as fraud: {result['is_fraud'].sum()}")
    print(f"\nFraud Summary:")
    print(get_fraud_summary(result))

    print(f"\nFlagged transactions:")
    print(result[result['is_fraud']][['sender', 'receiver', 'amount', 'fraud_type']])
