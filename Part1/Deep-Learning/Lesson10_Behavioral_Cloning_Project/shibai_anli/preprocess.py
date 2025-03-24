import numpy as np

import pandas as pd
from pandas import read_csv

balanced = pd.DataFrame()
bins = 1000
bin_n = 300

df = read_csv("../data/driving_log.csv")

start = 0
for end in np.linspace(0, 1, num=bins):
    df_range = df[(np.absolute(df.steering) >= start) & (np.absolute(df.steering) < end)]
    range_n = min(bin_n,df_range.shape[0])
    balanced = pd.concat([balanced,df_range.sample(range_n)])
    start = end

balanced.to_csv('../data/driving_log_balanced.csv',index=False)
