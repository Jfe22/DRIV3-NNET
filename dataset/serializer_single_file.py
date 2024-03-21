import pandas as pd

# Load CSV data into a pandas DataFrame
data = pd.read_csv("Acceleration/Till reached distance/10m/aggressive/BMW e36/acceleration 0m 10m aggressive-23-04-2023-19-42-30.csv")

# Display the first few rows of the DataFrame to inspect the data
print(data.head())
