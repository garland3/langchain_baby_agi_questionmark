
import os

try:
    X = data.drop('Temperature', axis=1)
except KeyError as e:
    if 'Temperature' in str(e):
        os.system('echo Temperature not found in data.csv > model_report.md')
    else:
        raise e
