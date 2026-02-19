import pandas as pd
import numpy as np

def generate_transaction_data(n_entities=1000, n_transactions=5000, fraud_ratio=0.05, seed=42):
    """
    Generates synthetic transaction data with potential fraud rings.
    """
    np.random.seed(seed)
    entities = [f"User_{i}" for i in range(n_entities)]
    
    # Base transactions
    df = pd.DataFrame({
        'sender': np.random.choice(entities, n_transactions),
        'receiver': np.random.choice(entities, n_transactions),
        'amount': np.round(np.random.exponential(scale=1000, size=n_transactions), 2),
        'timestamp': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 180, size=n_transactions), unit='D')
    })
    
    # Remove self-transfers
    df = df[df['sender'] != df['receiver']].reset_index(drop=True)
    
    # Initialize fraud labels
    df['is_fraud'] = False
    
    # Inject synthetic fraud rings (e.g., circular paths)
    n_fraud = int(n_transactions * fraud_ratio)
    fraud_indices = np.random.choice(df.index, n_fraud, replace=False)
    df.loc[fraud_indices, 'is_fraud'] = True
    
    # Specific pattern: A small set of nodes transacting heavily with each other
    fraud_ring_nodes = np.random.choice(entities, 5, replace=False)
    for i in range(len(fraud_ring_nodes)):
        sender = fraud_ring_nodes[i]
        receiver = fraud_ring_nodes[(i + 1) % len(fraud_ring_nodes)]
        new_row = {
            'sender': sender,
            'receiver': receiver,
            'amount': np.random.uniform(5000, 10000),
            'timestamp': pd.to_datetime('2023-03-15'),
            'is_fraud': True
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
    return df

def process_uploaded_csv(uploaded_file):
    """
    Processes an uploaded CSV file, mapping standard columns to internal format.
    Expected columns: sender id, receiver id, transaction amount, timestamps, [is_fraud]
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Column mapping dictionary
        mapping = {
            'sender id': 'sender',
            'receiver id': 'receiver',
            'transaction amount': 'amount',
            'timestamps': 'timestamp',
            'sender_id': 'sender',
            'receiver_id': 'receiver',
            'amount': 'amount',
            'timestamp': 'timestamp'
        }
        
        # Normalize columns (case-insensitive and strip whitespace)
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Apply mapping
        df = df.rename(columns=mapping)
        
        # Ensure required columns exist
        required = ['sender', 'receiver', 'amount', 'timestamp']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
            
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Handle is_fraud if missing
        if 'is_fraud' not in df.columns:
            df['is_fraud'] = False
        else:
            # Ensure boolean
            df['is_fraud'] = df['is_fraud'].astype(bool)
            
        return df
    except Exception as e:
        raise ValueError(f"Error processing CSV: {str(e)}")

if __name__ == "__main__":
    df = generate_transaction_data()
    print(df.head())
    print(f"Total transactions: {len(df)}")
    print(f"Fraudulent transactions: {df['is_fraud'].sum()}")
