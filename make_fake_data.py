# make some fake data
# 3 columns. x1, x2, x3,  temperature
# use a basic function to generate the temperature with some noise.temperature is a function of the other 3 columns
# save to data.csv
# make a plot of the data, save as data.png. Might rquire subplots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Define function for generating temperature with noise
def temp_func(x1, x2, x3):
    noise = np.random.normal(loc=0, scale=0.1)
    return 2*x1 + 3*x2 - 2*x3 + noise

# Generate fake data
data = pd.DataFrame({
    'x1': np.random.uniform(low=0, high=1, size=100),
    'x2': np.random.uniform(low=0, high=1, size=100),
    'x3': np.random.uniform(low=0, high=1, size=100)
})
data['temperature'] = data.apply(lambda row: temp_func(row['x1'], row['x2'], row['x3']), axis=1)

# Save data to CSV
data.to_csv('data.csv', index=False)

# Plot data
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].scatter(data['x1'], data['temperature'])
axs[0].set_xlabel('x1')
axs[0].set_ylabel('Temperature')

axs[1].scatter(data['x2'], data['temperature'])
axs[1].set_xlabel('x2')
axs[1].set_ylabel('Temperature')

axs[2].scatter(data['x3'], data['temperature'])
axs[2].set_xlabel('x3')
axs[2].set_ylabel('Temperature')

plt.tight_layout()
plt.savefig('data.png')
