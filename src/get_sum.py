import pandas as pd

def get_sum(source_file_path, output_file_path, file_name, start_month):
  data = pd.read_csv(source_file_path)
  data = data.set_index('Merchant Name')
  dates = pd.Series(pd.period_range(start_month, freq="M", periods=len(data.columns)))
  data.columns = dates
  data.sum(axis=0).to_csv(output_path+file_name+'.csv', header=False)
  return data.sum(axis=0)