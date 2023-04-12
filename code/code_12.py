# Import necessary libraries
import pandas as pd
import numpy as np
# Read in the data
data = pd.read_csv('data.csv')
# Check for any missing values
data.isnull().sum()
# Check for any outliers
data.describe()
# Check for any correlations
data.corr()