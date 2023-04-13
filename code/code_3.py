import pandas as pd
# Read the CSV file into a DataFrame
data = pd.read_csv("data.csv")
# Find missing values in the DataFrame
missing_values = data.isnull()
# Count the number of missing values in each column
num_missing = missing_values.sum()
# Calculate the percentage of missing values in each column
percent_missing = num_missing / len(data) * 100
# Create a table of the results
missing_data = pd.concat([num_missing, percent_missing], axis=1, keys=['Total', 'Percent'])
# Print the table
print(missing_data)