from sklearn.model_selection import GridSearchCV
# Define the parameters to search through
parameters = {'max_depth': [2, 4, 6, 8],
              'min_samples_split': [2, 4, 6, 8],
              'min_samples_leaf': [2, 4, 6, 8]}
# Create the grid search object
grid_search = GridSearchCV(model, parameters, cv=5)
# Fit the grid search to the data
grid_search.fit(X, y)
# Print the best parameters
print(grid_search.best_params_)