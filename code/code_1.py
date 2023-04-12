# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# read in data
df = pd.read_csv('weather_data.csv')
# create features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
# fit linear regression model
model = LinearRegression()
model.fit(X, y)
# make predictions
predictions = model.predict(X)
# write report
with open('weather_report.md', 'w') as f:
    f.write('# Weather Report for SF Today\n\n')
    f.write('Based on the previous 5 days of data, the weather in SF today is predicted to be: {}\n\n'.format(predictions[-1]))
    f.write('The following table shows the data used to make the prediction:\n\n')
    f.write(df.to_markdown())