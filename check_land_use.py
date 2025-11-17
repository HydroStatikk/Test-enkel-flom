import pandas as pd

# Load the Excel file
file_path = "attached_assets/Data.xlsx"
df = pd.read_excel(file_path)

# Print the column names to see what's available
print("Available columns:")
for col in df.columns:
    print(f"- {col}")

# Check for land use columns
land_use_columns = [
    "percentAgricul", 
    "percentBog", 
    "percentEffLake", 
    "percentForest", 
    "percentLake", 
    "percentMountain", 
    "percentUrban"
]

# Check which ones exist
for col in land_use_columns:
    if col in df.columns:
        print(f"✓ {col} is present")
    else:
        print(f"✗ {col} is missing")