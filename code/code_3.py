# Import necessary libraries
import pandas as pd
import numpy as np
# Read in the data from the Dark Sky API
df = pd.read_csv('weather_data.csv')
# Clean the data
# Remove any rows with missing values
df.dropna(inplace=True)
# Convert temperature from Fahrenheit to Celsius
df['temperature'] = (df['temperature'] - 32) * 5/9
# Convert wind speed from mph to m/s
df['wind_speed'] = df['wind_speed'] * 0.44704
# Convert precipitation from inches to millimeters
df['precipitation'] = df['precipitation'] * 25.4
# Convert pressure from inches of mercury to millibars
df['pressure'] = df['pressure'] * 33.86
# Write the cleaned data to a new csv file
df.to_csv('weather_data_cleaned.csv', index=False)
# Import necessary libraries
from sklearn.linear_model import LinearRegression
# Read in the cleaned data
df = pd.read_csv('weather_data_cleaned.csv')
# Create the X and y variables
X = df.drop(['temperature'], axis=1)
y = df['temperature']
# Create the linear regression model
model = LinearRegression()
model.fit(X, y)
# Make predictions
predictions = model.predict(X)