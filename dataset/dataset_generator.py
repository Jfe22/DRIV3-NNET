import pandas as pd

DEBUG = 0

concatenated_df = pd.DataFrame()
concatenated_rows = []

# Read aggressive datasets
for i in range(1, 11):
  df = pd.read_csv(f"../dataset/datasets_for_pandas/dataset_agg{i}.csv")
  if (DEBUG): print(f'dataframe: {df}')

  window_size = 4
  increment = 1
  for i in range(len(df) - window_size + 1):
    window = df.iloc[i:i + window_size]
    if (DEBUG): print("window: " + window) 
    
    concatenated_sensor_data = ' '.join(map(str, window['sensor_data']))
    label = window.iloc[0]['label'] 

    concatenated_rows.append({'sensor_data': concatenated_sensor_data, 'label': label})
    if (not DEBUG): print("sensor data: " + concatenated_sensor_data) 
    if (not DEBUG): print(f'concated rows: {concatenated_rows}') 

concatenated_df = pd.DataFrame(concatenated_rows) 
concatenated_df.to_csv('test_dataset1.csv', index=False)