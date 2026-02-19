from data_manager import generate_transaction_data
from graph_processor import build_graph, calculate_metrics
import pandas as pd

def test_workflow():
    print("Testing Synthetic Data Generation...")
    df = generate_transaction_data(n_entities=100, n_transactions=500, fraud_ratio=0.1)
    assert len(df) >= 500, "Dataframe size mismatch"
    assert df['is_fraud'].sum() > 0, "No fraud detected in simulation"
    print(f"Success: Generated {len(df)} transactions with {df['is_fraud'].sum()} fraud labels.")

    print("\nTesting Graph Processing...")
    G = build_graph(df)
    assert G.number_of_nodes() > 0, "Graph has no nodes"
    assert G.number_of_edges() > 0, "Graph has no edges"
    print(f"Success: Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    print("\nTesting Metric Calculation...")
    metrics_df = calculate_metrics(G)
    assert not metrics_df.empty, "Metrics dataframe is empty"
    assert 'pagerank' in metrics_df.columns, "PageRank metric missing"
    print("Success: Calculated graph metrics.")
    print(metrics_df.sort_values(by='pagerank', ascending=False).head())

if __name__ == "__main__":
    try:
        test_workflow()
        print("\nALL CORE TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
