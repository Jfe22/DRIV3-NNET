import pandas as pd
import random

DEBUG = 0
concatenated_rows = []
concatenated_dataset = []
window_size = 4
increment = 1


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


concatenate_dataset = pd.concat(concatenated_dataset)
concatenate_dataset.to_csv('datasets_for_training/tt_full_dataset.csv', index=False)

