import pandas as pd
# Read the CSV file into a DataFrame
data = pd.read_csv("data.csv")
# Drop rows with missing values in the 'column_name' column
column_name = 'columnName'
data = data.dropna(subset=[column_name])