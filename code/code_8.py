import pandas as pd
# Read the CSV file into a DataFrame
data = pd.read_csv("data.csv")
# Check the percentage of missing values in each column
missing_percent = (data.isna().sum() / len(data)) * 100
# Print the percentage of missing values for each column
print(missing_percent)