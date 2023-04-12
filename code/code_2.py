# Import necessary libraries
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
# Get data from Dark Sky API
url = 'https://api.darksky.net/forecast/[API KEY]/37.7749,-122.4194'
response = requests.get(url)
data = response.json()
# Create a dataframe with the data
df = pd.DataFrame(data['daily']['data'])
# Create a linear regression model
X = df[['temperatureHigh', 'temperatureLow', 'humidity', 'pressure', 'windSpeed']]
y = df['precipProbability']
model = LinearRegression().fit(X, y)
# Make a prediction for today
today_data = [data['currently']['temperatureHigh'], data['currently']['temperatureLow'], data['currently']['humidity'], data['currently']['pressure'], data['currently']['windSpeed']]
prediction = model.predict([today_data])
# Write the report to a file
with open('weather_report.md', 'w') as f:
    f.write('# Weather Report for SF Today\n\n')
    f.write('Today\'s forecast is for a high of {}°F and a low of {}°F. The humidity is {}%, the pressure is {}mb, and the wind speed is {}mph. The chance of precipitation is {}%.\n\n'.format(data['currently']['temperatureHigh'], data['currently']['temperatureLow'], data['currently']['humidity'], data['currently']['pressure'], data['currently']['windSpeed'], round(prediction[0]*100, 2)))
    f.write('The predictive model was created using the data from the past 5 days.\n\n')
    f.write('| Date | High | Low | Humidity | Pressure | Wind Speed | Precipitation |\n')
    f.write('|------|------|-----|----------|----------|------------|---------------|\n')
    for index, row in df.iterrows():
        f.write('| {} | {}°F | {}°F | {}% | {}mb | {}mph | {}% |\n'.format(index, row['temperatureHigh'], row['temperatureLow'], row['humidity'], row['pressure'], row['windSpeed'], round(row['precipProbability']*100, 2)))