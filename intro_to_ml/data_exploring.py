import pandas as pd

melb_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melb_file_path)
melbourne_data.describe()


iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)
description = home_data.describe()

avg_lot_size = round(description.loc['mean', 'LotArea'])
newest_home_age = 2026 - description.loc['max', 'YearBuilt']

print(avg_lot_size, newest_home_age)
