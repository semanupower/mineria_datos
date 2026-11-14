from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
import kagglehub
from pathlib import Path
import os

dir_actual = os.path.dirname(__file__)
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

# filtro de años a usarse en el analisis
df['year'] = pd.to_numeric(df['year'], errors='coerce')# se escogen los años 1908 y 2017 para tener un conjunto mas completo
df = df[(df['year'] >= 1908) & (df['year'] <= 2017)] # desde 1907 se encuentran varios lanzamientos registrados, 2020 no tiene lanzamientos completos

# conteo de formatos
# escoger formatos con arriba de 100 lanzamientos para filtrar los que no tienen tantos registros
n_formatos = df['format'].value_counts()
formatos_val = n_formatos[n_formatos >= 100].index
df_valid = df[df['format'].isin(formatos_val)]

#perfiles de cada formato
conteo = df_valid.groupby(['format', 'year'])['release_id'].count().reset_index(name='conteo')
def perfil(g: pd.DataFrame):
    years = g['year'].values
    pesos = g['conteo'].values
    media = np.average(years, weights=pesos)
    return pd.Series({'total': pesos.sum(),
                      'inicio': years.min(),
                      'pico': years[pesos.argmax()],
                      'año_medio': media,
                      'duracion': years.max() - years.min(),
                      'dispersion': np.sqrt(np.average((years-media)**2, weights=pesos))})

perfil_formato = conteo.groupby('format').apply(perfil, include_groups=False).reset_index()
# vinyl y CD dominan los lanzamientos, entonces las distancias de kmeans se hacen más grandes
# con un log10 para evitar distancias injustas y poder categorizar más facilmente
perfil_formato['log_total'] = np.log10(perfil_formato['total'])

# realizando k means con silhouette score y elbow method
X = perfil_formato[['log_total', 'inicio', 'pico', 'año_medio', 'dispersion']]
X_scaled = StandardScaler().fit_transform(X)

#pruebas que se hicieron con los mejores resultados. se escogieron 3 y 4
"""
resultados = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    resultados.append((k, km.inertia_, silhouette_score(X_scaled, km.labels_)))

for k, inercia, sil in resultados:
    print(f'k={k}, inercia={inercia:.2f}, silhouette={sil:.2f}') """

for k in [3,4]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    perfil_formato[f'cluster_k{k}'] = km.fit_predict(X_scaled)

    print(f"\n===== k = {k} =====")
    print(perfil_formato.groupby(f'cluster_k{k}')[
        ['total', 'inicio', 'pico', 'año_medio', 'dispersion']
    ].mean().round(1))
    print(perfil_formato[f'cluster_k{k}'].value_counts().sort_index())

for c in sorted(perfil_formato['cluster_k4'].unique()):
    formatos = perfil_formato[perfil_formato['cluster_k4'] == c].sort_values('total', ascending=False)['format'].tolist()
    print(f"Cluster {c}: {formatos}")

conteo_c = conteo.merge(perfil_formato[['format', 'cluster_k4']], on='format')

# se normaliza cada dato para que el vinilo no se apodere de la grafica
conteo_c['share'] = conteo_c['conteo'] / conteo_c.groupby('format')['conteo'].transform('max')

curvas = conteo_c.groupby(['cluster_k4', 'year'])['share'].median().reset_index()
plt.figure(figsize=(12, 6))
for c, g in curvas.groupby('cluster_k4'):
    plt.plot(g['year'], g['share'], label=f'Cluster {c}', linewidth=2)
plt.xlabel('Año')
plt.ylabel('Lanzamientos (relativo al pico de cada formato)')
plt.title('Ciclo de vida promedio por cluster')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(dir_actual, 'analisis_picos_clusters.png'))

# Empezando el analisis ANOVA
clusters = {0: 'Era digital', 1: 'Era temprana', 2: 'Era 90-00s', 3: 'Vinilo/análogos'}
mapa = perfil_formato.set_index('format')['cluster_k4'].map(clusters)
df_a = df_valid.copy()
df_a['cluster'] = df_a['format'].map(mapa)

etiquetas = sorted(df_a['cluster'].unique())
grupos = [df_a.loc[df_a['cluster'] == e, 'year'].values for e in etiquetas]

f_stat, p_anova = stats.f_oneway(*grupos)
print(f'F: {f_stat}, P: {p_anova}')
tukey = pairwise_tukeyhsd(endog=df_a['year'], groups=df_a['cluster'], alpha=0.05)
tabla = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
print(tabla, '\n') 

# forecasting.
formatos_vivos = ['CD', 'File', 'Cassette', 'DVD', 'SACD', 'Blu-ray', 'Vinyl', 'Box Set'] # se eligen los formatos actualmente usados para no contaminar el forecasting
def serie_formato(nombre) -> pd.DataFrame:
    s = (df_valid[df_valid['format'] == nombre].groupby('year')['release_id'].count().reindex(range(1908,2018), fill_value=0))
    return s

# realizando una funcion que implemente el WAPE
def wape(real, pred):
    return np.sum(np.abs(real-pred)) / np.sum(np.abs(real))

#reservar los ultimos 10 años como pruebas y el resto para entrenar
def validar(serie: pd.DataFrame, years_prueba = 10):
    train, test = serie.iloc[:-years_prueba], serie.iloc[-years_prueba:]
    modelo = ExponentialSmoothing(train.values, trend='add', damped_trend=True).fit()
    pred = np.clip(modelo.forecast(years_prueba),0, None)
    rmse = np.sqrt(np.mean((test.values - pred)**2))
    return wape(test.values, pred), rmse

#checando formatos que puedan ser quitados.
for f in formatos_vivos:
    val_wape, rmse = validar(serie_formato(f))
    print(f'{f}: WAPE = {val_wape:.2%}, RMSE: {rmse:.2f}')

# los resultados son prometedores para la elección, se entrena para hacer la prueba real
inicio, fin, objetivo = 1908, 2017, 2028
pasos = objetivo - fin                          
años_f = list(range(fin + 1, objetivo + 1))     # 2018-2028
fig, axes = plt.subplots(4,2, figsize=(14,14))
for ax, f in zip(axes.ravel(), formatos_vivos):
    s = serie_formato(f)
    modelo = ExponentialSmoothing(s.values, trend='add', damped_trend=True).fit()
    pred = np.clip(modelo.forecast(pasos), 0, None)

    ax.plot(s.index, s.values, label='Historico')
    ax.plot(list(años_f), pred, 'r--', marker='o', label='Pronostico')
    ax.set_xlim(1950,objetivo + 1)
    ax.set_title(f)
    ax.legend()

plt.suptitle('Pronóstico de lanzamientos por formato (hasta 2028)')
plt.tight_layout()
plt.savefig(os.path.join(dir_actual, 'forecast.png'), dpi=200)