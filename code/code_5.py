# Load the data
data = pd.read_csv('data.csv')
# Make predictions
predictions = model.predict(data)
# Print the predictions
print(predictions)