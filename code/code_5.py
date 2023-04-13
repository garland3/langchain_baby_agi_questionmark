import pandas as pd
# Read the CSV file into a DataFrame
data = pd.read_csv("data.csv")
# Print the list of column names to check if 'columnName' exists
print(data.columns)
# Update the column name in subset parameter
column_name = 'updatedColumnName'
data = data.dropna(subset=[column_name])