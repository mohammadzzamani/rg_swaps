import pandas as pd
import numpy as np

narrated_df = pd.read_csv('inputs/rg_swaps_with_narratives.csv',sep=',')
print (narrated_df.shape)
print (narrated_df.columns)