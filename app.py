import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from data_manager import process_uploaded_csv
from graph_processor import build_graph, calculate_metrics
from visualizer import create_network_viz
from fraud_detector import detect_all_fraud, get_fraud_summary, generate_json_report
import time

# Page config
st.set_page_config(page_title="Graph-Based Fraud Detection", layout="wide")

st.title("🛡️ Graph-Based Fraud Detection Dashboard")
st.markdown("""
This application uses **graph-based algorithms** to automatically detect fraudulent transaction patterns 
including **fan-out**, **fan-in**, **cycle**, and **chain** fraud — no pre-labeled data required.
""")

# Sidebar settings
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        st.session_state.df = process_uploaded_csv(uploaded_file)
        st.session_state.source = "upload"
        st.session_state.detected = False
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")

if 'df' not in st.session_state:
    st.warning("Please upload a CSV file to begin.")
    st.stop()

# --- Fraud Detection Engine ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Detection Settings")
time_window = st.sidebar.slider("Time Window (days)", 1, 14, 3, help="Window for fan-out/fan-in detection")
fanout_thresh = st.sidebar.slider("Fan-out Threshold", 3, 30, 10, help="Min unique receivers to flag fan-out")
fanin_thresh = st.sidebar.slider("Fan-in Threshold", 3, 30, 10, help="Min unique senders to flag fan-in")

run_detection = st.sidebar.button("🚀 Run Fraud Detection", type="primary")

df = st.session_state.df

if run_detection or not st.session_state.get('detected', False):
    with st.spinner("Running fraud detection algorithms..."):
        start_time = time.time()
        df = detect_all_fraud(df, time_window_days=time_window, 
                             fanout_threshold=fanout_thresh, fanin_threshold=fanin_thresh)
        processing_time = time.time() - start_time
        st.session_state.df = df
        st.session_state.processing_time = processing_time
        st.session_state.detected = True

# --- Color Legend ---
st.markdown("""
**Legend:** 🔴 Fan-out &nbsp;&nbsp; 🟠 Fan-in &nbsp;&nbsp; 🟣 Cycle &nbsp;&nbsp; 🟡 Chain &nbsp;&nbsp; 🔵 Legitimate
""")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Network Visualization")
    
    # Create and display visualization
    viz_path = create_network_viz(df)
    with open(viz_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    components.html(html_content, height=650)

with col2:
    st.subheader("🚨 Detection Results")
    
    # Fraud summary
    fraud_count = df['is_fraud'].sum()
    total_count = len(df)
    
    c1, c2 = st.columns(2)
    c1.metric("Total Transactions", total_count)
    c2.metric("Flagged as Fraud", int(fraud_count))
    
    if fraud_count > 0:
        # Breakdown by type
        summary = get_fraud_summary(df)
        st.markdown("**Fraud Breakdown by Type:**")
        st.dataframe(summary, hide_index=True, width='stretch')
        
        # Download JSON Report
        json_report = generate_json_report(df, st.session_state.get('processing_time', 0.0))
        st.download_button(
            label="📥 Download JSON Report",
            data=json_report,
            file_name="fraud_detection_report.json",
            mime="application/json"
        )
        
        # Flagged transactions detail
        st.markdown("**Flagged Transactions:**")
        fraud_df = df[df['is_fraud']][['sender', 'receiver', 'amount', 'fraud_type']].copy()
        fraud_df['amount'] = fraud_df['amount'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(fraud_df, hide_index=True, width='stretch')
    else:
        st.success("No fraud patterns detected in this dataset.")
    
    # Graph metrics
    st.subheader("📊 Graph Metrics")
    G = build_graph(df)
    metrics_df = calculate_metrics(G)
    
    # Show top nodes, prioritizing flagged ones
    flagged_nodes = metrics_df[metrics_df['is_fraud']].sort_values('pagerank', ascending=False)
    if not flagged_nodes.empty:
        st.markdown("**Suspicious Nodes (by PageRank):**")
        st.dataframe(flagged_nodes.head(10), hide_index=True, width='stretch')
    
    st.markdown("**Top 10 Nodes by PageRank:**")
    st.dataframe(metrics_df.sort_values(by='pagerank', ascending=False).head(10), hide_index=True, width='stretch')

# Detailed data view
with st.expander("View All Transaction Data"):
    st.dataframe(df, width='stretch')
