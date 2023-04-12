# Load data from data.csv
import pandas as pd
data = pd.read_csv('data.csv')
# Create machine learning model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(data[['x1', 'x2', 'x3']], data['y'])
# Make predictions on new data
predictions = model.predict(data[['x1', 'x2', 'x3']])
# Compare performance of model to other machine learning models
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(data['y'], predictions)
print('Mean Squared Error:', mse)