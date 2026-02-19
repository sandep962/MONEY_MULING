from data_manager import process_uploaded_csv
import os

def verify_csv_processing():
    csv_path = r"c:\Users\Jyoti Shetti\OneDrive\Desktop\rift\fraud_transactions.csv"
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Testing CSV processing with {csv_path}...")
    try:
        # Standard file object won't work for process_uploaded_csv if it expects a Streamlit UploadedFile (BytesIO)
        # but my process_uploaded_csv uses pd.read_csv(uploaded_file) which works with paths or file-like objects.
        df = process_uploaded_csv(csv_path)
        print("CSV Processed Successfully!")
        print(f"Rows: {len(df)}")
        print("Columns:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Verify required columns mapping
        required = ['sender', 'receiver', 'amount', 'timestamp']
        for col in required:
            if col not in df.columns:
                print(f"Error: Column {col} missing after mapping.")
                return
        
        print("\nALL CSV VERIFICATION TESTS PASSED!")
    except Exception as e:
        print(f"Verification Failed: {e}")

if __name__ == "__main__":
    verify_csv_processing()
