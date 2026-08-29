# clase 20220205 min 1:01:43
import kagglehub
import pandas as pd
import re
import os
from tabulate import tabulate

dir = f'{os.path.dirname(__file__)}\\dataset'
if os.path.isdir(f'{dir}\\datasets') == False:
    os.environ['KAGGLEHUB_CACHE'] = f'{dir}'
    path = kagglehub.dataset_download("sohrabdaemi/discogs-database-all-release-data")
else:
    path = f'{dir}\\datasets\\sohrabdaemi\\discogs-database-all-release-data\\versions\\1'

df = pd.read_csv(f'{path}\\release_data\\release_data.csv')
print(tabulate(df.head(10), headers=df.columns, tablefmt='orgtbl'))