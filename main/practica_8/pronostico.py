import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
import kagglehub
from pathlib import Path
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def get_csv() -> str:
    dir = f'{Path(__file__).parents[1]}\\dataset'
    if os.path.isdir(f'{dir}\\datasets') == False:
        os.environ['KAGGLEHUB_CACHE'] = f'{dir}'
        path = kagglehub.dataset_download("sohrabdaemi/discogs-database-all-release-data")
        return path
        
    path = f'{dir}\\datasets\\sohrabdaemi\\discogs-database-all-release-data\\versions\\1\\release_data\\release_data.csv'
    return path
csv = get_csv()
df = pd.read_csv(csv)

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df[(df['year'] >= 1930) & (df['year'] <= 2019)]
count = df.groupby(['genre', 'year'])['release_id'].count().reset_index(name='conteo')

generos = df['genre'].value_counts().head(6).index
print(generos)
for genero in generos:
    serie = (count[count['genre'] == genero].set_index('year')['conteo'].reindex(range(1930,2019), fill_value=0))
    modelo = ExponentialSmoothing(serie.values, trend='add').fit()
    pronostico = modelo.forecast(5)
    años_futuros = list(range(2020, 2025))

    linea, = plt.plot(serie.index, serie.values, label='genero')
    plt.plot(años_futuros, pronostico, '--', color=linea.get_color())

plt.xlabel('Año')
plt.ylabel('Lanzamientos')
plt.title('Pronóstico de lanzamientos por genero')
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()