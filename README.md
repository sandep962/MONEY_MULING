# Graph-Based Fraud Detection System

A powerful, modular Python application that detects complex financial fraud rings using relational graph algorithms. Unlike traditional machine learning models that rely on isolated, tabular features, this engine analyzes the *structural flow of money* to catch coordinated evasion tactics like laundering chains and circular flows.

##  Key Features

*   **Zero-Label Detection**: Automatically detects fraud patterns in raw, unlabeled transaction data. No pre-trained model or `is_fraud` labels required.
*   **Four Algorithmic Detection Engines**:
    *   🔴 **Fan-Out (Dispersion)**: One source distributing funds to many receivers in a short burst.
    *   🟠 **Fan-In (Funnel)**: Many sources consolidating funds into a single receiver in a short burst.
    *   🟣 **Cycle (Circular Flow)**: Money moving in a closed loop (A → B → C → A) to obscure origins or inflate volume.
    *   🟡 **Chain (Layering)**: Money moving through a long sequence of intermediary accounts with slightly decaying amounts.
*   **Interactive Graph Visualization**: Fully interactive 3D physics-based network graphs (powered by `pyvis`) to visually explore the detected fraud rings.
*   **Streamlit Dashboard**: A user-friendly web interface for uploading CSV data, tuning detection thresholds, and reviewing results.
*   **Compliance-Ready JSON Export**: Generates structured JSON reports containing extracted fraud rings, suspicion scores, and member accounts for downstream IT systems.

---

##  Architecture

The system is broken down into five core modules:

1.  **`app.py`**: The Streamlit dashboard. Connects the UI to the underlying engines.
2.  **`fraud_detector.py`**: The algorithmic brains. Contains the time-window sliding algorithms, DFS pathfinding, and NetworkX cycle detection to identify the 4 key fraud patterns.
3.  **`graph_processor.py`**: Calculates mathematical network metrics (like PageRank, In/Out degrees) and constructs the core NetworkX `MultiDiGraph`.
4.  **`visualizer.py`**: Translates the mathematical graph into a color-coded, interactive HTML visualization.
5.  **`data_manager.py`**: Handles CSV ingestion, column normalization, and data type formatting.

---

##  Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. You will need the following libraries:

```bash
pip install streamlit pandas networkx pyvis
```

### Running the Application

1. Clone or download this repository.
2. Navigate to the project directory in your terminal:
   ```bash
   cd path/to/project
   ```
3. Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
4. The application will open in your default web browser (usually at `http://localhost:8501`).

---

##  Using the Dashboard

1. **Upload Data**: Use the sidebar to upload a CSV file containing transaction data. 
   *(Required columns: `sender`, `receiver`, `amount`, `timestamp`)*.
2. **Tune Settings**: Adjust the time horizons and fan-in/fan-out thresholds depending on your dataset's specific volume.
3. **Run Detection**: Click "🚀 Run Fraud Detection".
4. **Analyze Results**: 
   * Review the highly suspicious nodes by `PageRank`.
   * Explore the interactive graph (red/orange/purple/yellow nodes indicate flagged fraud).
   * Download the JSON Compliance Report for downstream integration.

---

##  Example CSV Format

The system accepts CSV files loosely mapping to standard transaction logs. Example format:

| sender_id  | receiver_id | amount | timestamp           |
|------------|-------------|--------|---------------------|
| CUST_001   | MERCH_55    | 150.00 | 2025-08-01 09:00:00 |
| FRAUD_A    | MULE_1      | 5000.0 | 2025-08-01 09:05:00 |
| FRAUD_A    | MULE_2      | 4800.0 | 2025-08-01 09:06:00 |

*(The `data_manager.py` module automatically normalizes column names like `sender id` -> `sender`.)*

