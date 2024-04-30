import pandas as pd
import random

DEBUG = 0
concatenated_rows = []
concatenated_dataset = []
variance_dataset = []
window_size = 4
increment = 1



# 3 opcoes distintas para representar os novos dados
  # - Soma de valores positivos
  # - Arrays com os 4 valores
  # - Variancia de cada sensor
def get_arrays(window):
  print(window)
  for i in window["accelerometerXAxis"]:
    print(i)

  arrayAccX = [i for i in window["accelerometerXAxis"]]
  arrayAccY = [i for i in window["accelerometerYAxis"]]
  arrayAccZ = [i for i in window["accelerometerZAxis"]]
  arrayGyroX = [i for i in window["gyroscopeXAxis"]]
  arrayGyroY = [i for i in window["gyroscopeYAxis"]]
  arrayGyroZ = [i for i in window["gyroscopeZAxis"]]

  print(arrayAccX)
  print(arrayAccY)
  print(arrayAccZ)
  print(arrayGyroX)
  print(arrayGyroY)
  print(arrayGyroZ)


def get_variance(window):
  return



def get_windows(data):
  for i in range(len(data) - window_size + increment):
    window = data[i:i+window_size]
    get_arrays(window)
    exit()


# Read all data files and concat them
for i in range(1, 11):
  df = pd.read_csv(f"datasets_for_pandas/tt_dataset_agg{i}.csv")
  if DEBUG: print(f'dataframe agg{i}: {df}')
  concatenated_dataset.append(df)

  df = pd.read_csv(f"datasets_for_pandas/tt_dataset_normal{i}.csv")
  if DEBUG: print(f'dataframe normal{i}: {df}')
  concatenated_dataset.append(df)

  df = pd.read_csv(f"datasets_for_pandas/tt_dataset_slow{i}.csv")
  if DEBUG: print(f'dataframe slow{i}: {df}')
  concatenated_dataset.append(df)


#get_window(concatenated_dataset)


# -------------- uncomment when ready ---------------
concatenated_dataset = pd.concat(concatenated_dataset)
get_windows(concatenated_dataset)
#concatenate_dataset.to_csv('datasets_for_training/tt_full_dataset.csv', index=False)

