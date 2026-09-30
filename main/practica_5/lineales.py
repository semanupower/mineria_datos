import matplotlib.pyplot as plt
import os
from pathlib import Path
import kagglehub
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
dir_actual = os.path.dirname(__file__)

generos = df['genre'].unique()
for genero in generos:
    df_generos = df[df['genre'] == genero]
    generos_peryear = df_generos.groupby('year')['release_id'].count().reset_index()

    x = generos_peryear[['year']]
    y = generos_peryear['release_id']

    linear = LinearRegression()
    linear.fit(x,y)
    y_pred = linear.predict(x)
    r_2 = r2_score(y, y_pred)

    plt.figure(figsize=(10,5))
    plt.scatter(x,y, color='blue', s=100, label=f'Lanzamientos del género {genero}')
    plt.plot(x, y_pred, color='red', linewidth=3, label='Tendencia Lineal')
    plt.title(f'Análisis R^2 del género {genero} (R^2 = {r_2:.2f})')
    plt.xlabel('Año')
    plt.ylabel('Lanzamientos promedio')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    ruta = os.path.join(dir_actual, f'{genero.replace(' / ', '_')}')
    print(ruta)
    plt.savefig(ruta)
    plt.close()