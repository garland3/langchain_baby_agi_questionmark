# Import necessary libraries
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
# Read in data
data = pd.read_csv('data.csv')
# Split data into features and target
X = data.drop('temperature', axis=1)
y = data['temperature']
# Create a Random Forest Regressor
rf = RandomForestRegressor()
# Create a dictionary of hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6]
}
# Use GridSearchCV to tune the hyperparameters
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)
# Retrain the model with the optimal hyperparameters
rf_optimal = grid_search.best_estimator_
# Evaluate the performance of the model
score = rf_optimal.score(X, y)
print('Model performance: {}'.format(score))