import kagglehub
import os
from pathlib import Path
import pandas as pd
from tabulate import tabulate

def get_csv() -> str:
    dir = f'{Path(__file__).parents[1]}\\dataset'
    if os.path.isdir(f'{dir}\\datasets') == False:
        os.environ['KAGGLEHUB_CACHE'] = f'{dir}'
        path = kagglehub.dataset_download("sohrabdaemi/discogs-database-all-release-data")
        return path
        
    path = f'{dir}\\datasets\\sohrabdaemi\\discogs-database-all-release-data\\versions\\1\\release_data\\release_data.csv'
    return path

def mod_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(tabulate(df[df.isnull().any(axis=1)].head(10), headers=df.columns))
    df.dropna(inplace=True)
    df.to_csv(path, index=False)

path= get_csv()
mod_csv(path)
