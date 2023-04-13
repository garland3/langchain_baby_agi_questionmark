import pandas as pd
# Read the CSV file into a DataFrame
data = pd.read_csv("data.csv")
# Drop rows with missing values in the 'updatedColumnName' column
column_name = 'updatedColumnName'
data = data.dropna(subset=[column_name])