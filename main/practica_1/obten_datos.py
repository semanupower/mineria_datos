# clase 20220205 min 1:01:43
import kagglehub
import os
import discogs_client as discogs
from bs4 import BeautifulSoup
import requests
import pandas as pd

def get_csv() -> str:
    dir = f'{os.path.dirname(__file__)}\\dataset'
    if os.path.isdir(f'{dir}\\datasets') == False:
        os.environ['KAGGLEHUB_CACHE'] = f'{dir}'
        return path = kagglehub.dataset_download("sohrabdaemi/discogs-database-all-release-data")
        
    path = f'{dir}\\datasets\\sohrabdaemi\\discogs-database-all-release-data\\versions\\1'
    return path

def get_df(path: str):
    return pd.DataFrame(path)