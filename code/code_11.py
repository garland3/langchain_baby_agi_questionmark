
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = pd.read_csv('data.csv')

if os.path.exists('data.csv'):
    plt.plot(data['time'], data['accuracy'])
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.show()
    sns.barplot(x=data['time'], y=data['precision'], label='Precision')
    sns.barplot(x=data['time'], y=data['recall'], label='Recall')
    plt.title('Model Precision and Recall Scores')
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
    plt.plot(data['time'], data['mse'])
    plt.title('Model Mean Squared Error')
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.show()