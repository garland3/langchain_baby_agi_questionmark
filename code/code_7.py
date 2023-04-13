import pandas as pd
# Read the CSV file into a DataFrame
data = pd.read_csv("data.csv")
column_name = 'updatedColumnName'
# Check if the column exists in the DataFrame
if column_name in data.columns:
    # Drop rows with missing values in the column
    data = data.dropna(subset=[column_name])
else:
    print(f"{column_name} does not exist in the DataFrame.")