
try:
    plt.plot(data['time'], data['accuracy'])
except KeyError as err:
    print(f"KeyError: {err}")
    time_values = data.columns.values
    plt.plot(time_values, data['accuracy'])