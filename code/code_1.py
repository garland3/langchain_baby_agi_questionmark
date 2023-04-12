import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('data.csv')
parameters = {
    'n_estimators': [10, 50, 100],
    'max_depth': [2, 5, 10]
}
model = RandomForestRegressor()
clf = GridSearchCV(model, parameters, cv=5)
clf.fit(data.drop('temperature', axis=1), data['temperature'])
# Evaluate the model
score = clf.score(data.drop('temperature', axis=1), data['temperature'])
# Write the report
with open('model_report.md', 'w') as f:
    f.write('# Model Report\n\n')
    f.write('The model was tuned using hyperparameter tuning with the following parameters:\n\n')
    f.write('- n_estimators: 10, 50, 100\n')
    f.write('- max_depth: 2, 5, 10\n\n')
    f.write('The model achieved an accuracy score of {}.'.format(score))