from pathlib import Path
import pandas as pd

def get_csv() -> str:
    dir = f'{Path(__file__).parents[1]}\\dataset'
    path = f'{dir}\\datasets\\sohrabdaemi\\discogs-database-all-release-data\\versions\\1\\release_data\\release_data.csv'
    return path

path = get_csv()
df = pd.read_csv(path) 

print("Estadísticas descriptivas sobre los años de los lanzamientos")
tldr = df['year'].describe()
print(tldr)

print("\nLos 10 países con la mayor cantidad de lanzamientos")
top10 = df.groupby('country')['release_id'].count().sort_values(ascending=False).head(10)
print(top10)

print("\nCantidad promedio de lanzamientos por género de música")
prom_generos = df.groupby(['genre', 'format'])['release_id'].count().groupby('genre').mean().sort_values(ascending=False)
print(prom_generos)

print("\nRelease ID más reciente")
max_id = df['release_id'].max()
print(max_id)