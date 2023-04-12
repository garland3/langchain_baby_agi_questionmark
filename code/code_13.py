# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
# Read in the data
data = pd.read_csv('data.csv')
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('temperature', axis=1),
                                                    data['temperature'],
                                                    test_size=0.2,
                                                    random_state=0)
# Create the model
model = LinearRegression()
# Train the model
model.fit(X_train, y_train)
# Evaluate the model
accuracy = model.score(X_test, y_test)
# Print the accuracy
print('Accuracy: ', accuracy)