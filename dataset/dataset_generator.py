import pandas as pd
import random

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

  if DEBUG: print (f'window: {window}')

  for i in window:
    row = i.split("/")
    if DEBUG: print (f'row: {row}')
    accX.append(float(row[0].replace('$$$', '')))
    accY.append(float(row[1]))
    accZ.append(float(row[2]))
    gyroX.append(float(row[3]))
    gyroY.append(float(row[4]))
    gyroZ.append(float(row[5]))

  return accX, accY, accZ, gyroX, gyroY, gyroZ


def parse_sensor_data_from_string(string):
  accX = []
  accY = []
  accZ = []
  gyroX = []
  gyroY = []
  gyroZ = []
  
  window = string.split(" ")
  for i in window:
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
  if (driving_variance >= 0.6): label = 'Aggressive'
  return label 

def create_window(accX, accY, accZ, gyroX, gyroY, gyroZ):
  concatenated_data = []
  for i in range(window_size): concatenated_data.append(f'$$${accX[i]}/{accY[i]}/{accZ[i]}/{gyroX[i]}/{gyroY[i]}/{gyroZ[i]}')
  return concatenated_data

def extra_data(window):
  accX, accY, accZ, gyroX, gyroY, gyroZ = parse_sensor_data(window)
  random_value = random.uniform(-0.01, 0.01)
  newAccX = [x + random_value for x in accX]
  newAccY = [x + random_value for x in accY]
  newAccZ = [x + random_value for x in accZ]
  newGyroX = [x + random_value for x in gyroX]
  newGyroY = [x + random_value for x in gyroY]
  newGyroZ = [x + random_value for x in gyroZ]

  concatenated_data = create_window(newAccX, newAccY, newAccZ, newGyroX, newGyroY, newGyroZ)
  column_name = "sensor_data"
  concatenated_data_final = pd.DataFrame({column_name: concatenated_data})
  label = set_label(concatenated_data_final['sensor_data']) 

  return concatenated_data_final, label

def balance_dataset(df):
  aggressive_count = 0
  normal_count = 0
  slow_count = 0

  for i, row in df.iterrows():
    if (row['label'] == 'Aggressive'): aggressive_count += 1
    if (row['label'] == 'Normal'): normal_count += 1
    if (row['label'] == 'Slow'): slow_count += 1

  #WE ALWAYS HAVE MORE NORMAL VALUES
  agg_missing = normal_count - aggressive_count
  slow_missing = normal_count - slow_count

  while (agg_missing > 0):
    for i, row in df.iterrows():
      if (row['label'] == 'Aggressive'): 
        accX, accY, accZ, gyroX, gyroY, gyroZ = parse_sensor_data_from_string(row['sensor_data'])
        new_data_window = create_window(accX, accY, accZ, gyroX, gyroY, gyroZ)
        new_data_df = pd.DataFrame({'sensor_data': new_data_window})

        new_agg_data, label = extra_data(new_data_df['sensor_data'])
        concatenated_sensor_data = ' '.join(map(str, new_agg_data['sensor_data']))
        concatenated_rows.append({'sensor_data': concatenated_sensor_data, 'label': label})
        agg_missing -= 1
  
  while (slow_missing > 0):
    for i, row in df.iterrows():
      if (row['label'] == 'Slow'): 
        accX, accY, accZ, gyroX, gyroY, gyroZ = parse_sensor_data_from_string(row['sensor_data'])
        new_data_window = create_window(accX, accY, accZ, gyroX, gyroY, gyroZ)
        new_data_df = pd.DataFrame({'sensor_data': new_data_window})

        new_slow_data, label = extra_data(new_data_df['sensor_data'])
        concatenated_sensor_data = ' '.join(map(str, new_slow_data['sensor_data']))
        concatenated_rows.append({'sensor_data': concatenated_sensor_data, 'label': label})
        slow_missing -= 1


def concatenate_dataset(df, window_size, increment, concatenated_rows):
  for i in range(len(df) - window_size + increment):
    window = df.iloc[i:i + window_size]
    if DEBUG: print("window: " + window) 
    
    concatenated_sensor_data = ' '.join(map(str, window['sensor_data']))
    label = set_label(window['sensor_data'])

    concatenated_rows.append({'sensor_data': concatenated_sensor_data, 'label': label})
    if DEBUG: print("sensor data: " + concatenated_sensor_data) 
    if DEBUG: print(f'concated rows: {concatenated_rows}')

    #data augmentation - x2
    dataaug1_sensor_data, dataaug1_label = extra_data(window['sensor_data'])
    dataaug1_concatenated_sensor_data = ' '.join(map(str, dataaug1_sensor_data['sensor_data']))

    concatenated_rows.append({'sensor_data': dataaug1_concatenated_sensor_data, 'label': dataaug1_label})
    if DEBUG: print(f'concated rows: {concatenated_rows}')

for i in range(1, 11):
  df = pd.read_csv(f"datasets_for_pandas/dataset_agg{i}.csv")
  if DEBUG: print(f'dataframe agg{i}: {df}')
  concatenate_dataset(df, window_size, increment, concatenated_rows)

  df = pd.read_csv(f"datasets_for_pandas/dataset_normal{i}.csv")
  if DEBUG: print(f'dataframe normal{i}: {df}')
  concatenate_dataset(df, window_size, increment, concatenated_rows)

  df = pd.read_csv(f"datasets_for_pandas/dataset_slow{i}.csv")
  if DEBUG: print(f'dataframe slow{i}: {df}')
  concatenate_dataset(df, window_size, increment, concatenated_rows)

#balance_dataset(concatenated_rows)
concatenated_df = pd.DataFrame(concatenated_rows) 
balance_dataset(concatenated_df)
concatenated_balanced_df = pd.DataFrame(concatenated_rows)

#concatenated_df.to_csv('datasets_for_training/auto_label_data_augmented_2.csv', index=False)
concatenated_balanced_df.to_csv('datasets_for_training/auto_label_data_augmented_2.csv', index=False)
