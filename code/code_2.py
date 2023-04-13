import pandas as pd
# Read the CSV file into a DataFrame
data = pd.read_csv("data.csv")
# Identify missing values in the DataFrame
missing_values = data.isnull()
# Count the number of missing values in each column
num_missing = missing_values.sum()
# Print the number of missing values in each column
print(num_missing)