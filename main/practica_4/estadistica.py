from statsmodels.stats.multicomp import pairwise_tukeyhsd
import kagglehub
from pathlib import Path
from scipy import stats
import pandas as pd
import os

def get_csv() -> str:
    dir = f'{Path(__file__).parents[1]}\\dataset'
    if os.path.isdir(f'{dir}\\datasets') == False:
        os.environ['KAGGLEHUB_CACHE'] = f'{dir}'
        path = kagglehub.dataset_download("sohrabdaemi/discogs-database-all-release-data")
        return path
        
    path = f'{dir}\\datasets\\sohrabdaemi\\discogs-database-all-release-data\\versions\\1\\release_data\\release_data.csv'
    return path
path = get_csv()
df = pd.read_csv(path)

print("Promedio de lanzamientos por año de cada género")
prom_generos = df.groupby(['genre', 'year'])['release_id'].count().groupby('genre').mean().sort_values(ascending=False)
print(f"{prom_generos}\n")

print("Prueba ANOVA")
tot_generos = df.groupby(['genre', 'year'])['release_id'].count().reset_index(name='total')
generos = df['genre'].unique()
grupos = [valores['total'].values for _, valores in tot_generos.groupby('genre')]
F, p_val = stats.f_oneway(*grupos)
print(f"Estadistico F = {F}\nP-valor = {p_val}")

if p_val < 0.05:
    print("Hay diferencias significativas entre uno o más géneros.")
else:
    print("No hay diferencias significativas.")

tukey = pairwise_tukeyhsd(endog=tot_generos['total'], groups= tot_generos['genre'], alpha=0.05)
print(tukey)