# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Read in the data
data = pd.read_csv('data.csv')
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Temperature', axis=1), data['Temperature'], test_size=0.2, random_state=42)
# Create the model
model = LinearRegression()
model.fit(X_train, y_train)
# Evaluate the model
score = model.score(X_test, y_test)