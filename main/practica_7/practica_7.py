import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
import kagglehub
from pathlib import Path
import os
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

datos = df.groupby(['format', 'year'])['release_id'].count().reset_index(name='conteo')
x = datos[['year', 'conteo']]
x_scaler = StandardScaler().fit_transform(x)
k_means = KMeans(n_clusters=6, random_state=42, n_init=10)
datos['cluster'] = k_means.fit_predict(x_scaler)

reporte = (datos.groupby('format')['cluster'].agg(lambda x: x.mode()[0]).reset_index(name='cluster_format'))
reporte['total_lanzamientos'] = reporte['format'].map(datos.groupby('format')['conteo'].sum())
reporte = reporte.sort_values(by=['cluster_format', 'format'])

print(reporte.to_string(index=False))
reporte.to_csv(f'{os.path.dirname(__file__)}/reporte_discogs_formatos.csv', index=False, encoding='utf-8-sig')