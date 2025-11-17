import pandas as pd

# Load the Excel file
df = pd.read_excel("attached_assets/Data.xlsx")

# Print the column names
print("Column names:", df.columns.tolist())

# Print first row as example
print("\nFirst row example:")
print(df.iloc[0])