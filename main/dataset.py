# clase 20220205 min 1:01:43
import kagglehub
import pandas as pd
import re

# Download latest version
path = kagglehub.dataset_download("sohrabdaemi/discogs-database-all-release-data")

print("Path to dataset files:", path) 