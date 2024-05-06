import pandas as pd
import random

DEBUG = 0
concatenated_rows = []
concatenated_raw_dataset = []
variance_dataset = []
window_size = 4
increment = 1

def get_arrays(window):
  arrayAccX = [i for i in window["accelerometerXAxis"]]
  arrayAccY = [i for i in window["accelerometerYAxis"]]
  arrayAccZ = [i for i in window["accelerometerZAxis"]]
  arrayGyroX = [i for i in window["gyroscopeXAxis"]]
  arrayGyroY = [i for i in window["gyroscopeYAxis"]]
  arrayGyroZ = [i for i in window["gyroscopeZAxis"]]
  arrayLabel = [i for i in window["label"]]
  label = arrayLabel[0]
  varianceAccY = max(arrayAccY) - min(arrayAccY)

  return arrayAccX, arrayAccY, arrayAccZ, arrayGyroX, arrayGyroY, arrayGyroZ, label 

def set_label_mean(accY):
  nVals = 0
  sumAccY = 0

  for val in accY:
    nVals += 1
    sumAccY += val 
  meanAccY = sumAccY / nVals

  if (meanAccY > 0.8): return "Aggressive"
  return "Normal"

def set_label_variance(accY):
  varAccY = max(accY) - min(accY)

  if (varAccY > 0.8): return "Aggressive"
  return "Normal"

def get_windows(data):
  concatenated_4instarray_dataset = pd.DataFrame() 
  for i in range(len(data) - window_size + increment):
    window = data[i:i+window_size]
    accX, accY, accZ, gyroX, gyroY, gyroZ, label = get_arrays(window)
    label = set_label_variance(accY)

    new_row = pd.DataFrame({'accelerometerXAxis': [accX], 'accelerometerYAxis': [accY],  'accelerometerZAxis': [accZ],
                            'gyroscopeXAxis': [gyroX], 'gyroscopeYAxis': [gyroY], 'gyroscopeZAxis': [gyroZ], 
                            'label': [label]}, index=[0])
    concatenated_4instarray_dataset = pd.concat([concatenated_4instarray_dataset, new_row], ignore_index=True)

    if DEBUG: print(concatenated_4instarray_dataset)
  return concatenated_4instarray_dataset 

# Read all data files and concat them
for i in range(1, 11):
  df = pd.read_csv(f"datasets_for_pandas/tt_dataset_agg{i}.csv")
  if DEBUG: print(f'dataframe agg{i}: {df}')
  concatenated_raw_dataset.append(df)

  df = pd.read_csv(f"datasets_for_pandas/tt_dataset_normal{i}.csv")
  if DEBUG: print(f'dataframe normal{i}: {df}')
  concatenated_raw_dataset.append(df)

  f = pd.read_csv(f"datasets_for_pandas/tt_dataset_slow{i}.csv")
  if DEBUG: print(f'dataframe slow{i}: {df}')
  concatenated_raw_dataset.append(df)

# raw dataset concatenated
concatenated_raw_dataset = pd.concat(concatenated_raw_dataset)

concatenated_handled_dataset = get_windows(concatenated_raw_dataset)
print(concatenated_handled_dataset)

concatenated_handled_dataset.to_csv('datasets_for_training/tt_auto_labeled_variance_arrays_dataset.csv', index=False)
