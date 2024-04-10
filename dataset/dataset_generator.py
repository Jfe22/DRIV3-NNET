import pandas as pd

DEBUG = 1
concatenated_rows = []
window_size = 4
increment = 1

def concatenate_dataset(df, window_size, increment, concatenated_rows):
  for i in range(len(df) - window_size + increment):
    window = df.iloc[i:i + window_size]
    if (DEBUG): print("window: " + window) 
    
    concatenated_sensor_data = ' '.join(map(str, window['sensor_data']))
    label = window.iloc[0]['label'] 

    concatenated_rows.append({'sensor_data': concatenated_sensor_data, 'label': label})
    if (DEBUG): print("sensor data: " + concatenated_sensor_data) 
    if (DEBUG): print(f'concated rows: {concatenated_rows}')

for i in range(1, 11):
  df = pd.read_csv(f"datasets_for_pandas/dataset_agg{i}.csv")
  if (DEBUG): print(f'dataframe agg{i}: {df}')
  concatenate_dataset(df, window_size, increment, concatenated_rows)

  df = pd.read_csv(f"datasets_for_pandas/dataset_normal{i}.csv")
  if (DEBUG): print(f'dataframe normal{i}: {df}')
  concatenate_dataset(df, window_size, increment, concatenated_rows)

  df = pd.read_csv(f"datasets_for_pandas/dataset_slow{i}.csv")
  if (DEBUG): print(f'dataframe slow{i}: {df}')
  concatenate_dataset(df, window_size, increment, concatenated_rows)

concatenated_df = pd.DataFrame(concatenated_rows) 
concatenated_df.to_csv('datasets_for_training/1sec_timeframe_dataset.csv', index=False)