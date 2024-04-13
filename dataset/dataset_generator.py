import pandas as pd

DEBUG = 0
concatenated_rows = []
window_size = 4
increment = 1

def parse_sensor_data(window):
  accX = []
  accY = []
  accZ = []
  gyroX = []
  gyroY = []
  gyroZ = []
    
  for i in window['sensor_data']:
    row = i.split("/")
    accX.append(float(row[0].replace('$$$', '')))
    accY.append(float(row[1]))
    accZ.append(float(row[2]))
    gyroX.append(float(row[3]))
    gyroY.append(float(row[4]))
    gyroZ.append(float(row[5]))

  return accX, accY, accZ, gyroX, gyroY, gyroZ

def set_label(window):
  accX, accY, accZ, gyroX, gyroY, gyroZ = parse_sensor_data(window)
  accXVariance = max(accX) - min(accX) 
  accYVariance = max(accY) - min(accY)
  accZVariance = max(accZ) - min(accZ)
  gyroXVariance = max(gyroX) - min(gyroX)
  gyroYVariance = max(gyroY) - min(gyroY)
  gyroZVariance = max(gyroZ) - min(gyroZ)

  weight_accX = 1/6
  weight_accY = 1/6
  weight_accZ = 1/6
  weight_gyroX = 1/6
  weight_gyroY = 1/6
  weight_gyroZ = 1/6

  driving_variance = (accXVariance * weight_accX) + (accYVariance * weight_accY) + (accZVariance * weight_accZ) + (gyroXVariance * weight_gyroX) + (gyroYVariance * weight_gyroY) + (gyroZVariance * weight_gyroZ)

  if (driving_variance < 0.3): label = 'Slow'
  if (driving_variance >= 0.3 and driving_variance < 0.6): label = 'Normal'
  if (driving_variance >= 0.6): label = 'Aggresive'
  return label 

def concatenate_dataset(df, window_size, increment, concatenated_rows):
  for i in range(len(df) - window_size + increment):
    window = df.iloc[i:i + window_size]
    if (DEBUG): print("window: " + window) 
    
    concatenated_sensor_data = ' '.join(map(str, window['sensor_data']))
    #label = window.iloc[0]['label'] 
    label = set_label(window)

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
#concatenated_df.to_csv('datasets_for_training/1sec_timeframe_dataset.csv', index=False)
concatenated_df.to_csv('datasets_for_training/auto_label_test1.csv', index=False)