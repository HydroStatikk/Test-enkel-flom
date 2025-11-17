import pandas as pd

# Load the Excel file
try:
    df = pd.read_excel("attached_assets/Data.xlsx")
    print("Successfully loaded the file")
    print(f"Number of rows: {len(df)}")
    print("Columns in the dataframe:")
    for col in df.columns:
        print(f"- {col}")
    
    # Print the first few rows
    print("\nFirst 5 rows of data:")
    print(df.head().to_string())
    
except Exception as e:
    print(f"Error loading the file: {e}")