from matplotlib.ticker import ScalarFormatter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
import kagglehub

def get_csv() -> str:
    dir = f'{Path(__file__).parents[1]}\\dataset'
    if os.path.isdir(f'{dir}\\datasets') == False:
        os.environ['KAGGLEHUB_CACHE'] = f'{dir}'
        path = kagglehub.dataset_download("sohrabdaemi/discogs-database-all-release-data")
        return path
        
    path = f'{dir}\\datasets\\sohrabdaemi\\discogs-database-all-release-data\\versions\\1\\release_data\\release_data.csv'
    return path

dir_actual = os.path.dirname(__file__)
csv = get_csv()
df = pd.read_csv(csv)

# dispersion
sample = df.sample(10000)
plt.figure(figsize=(14,8))
sns.scatterplot(data=sample, x='year', y='release_id', hue="genre", alpha=0.5, sizes=(20,200))
plt.legend(fontsize=8, markerscale=1, labelspacing=0.3)
plt.title("Lanzamientos a través de los años")
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(dir_actual, "dispersion_release_year.jpg"))
plt.close()

# barras
plt.figure(figsize=(15,5))
top10 = df.groupby('country')['release_id'].count().sort_values(ascending=False).head(10)
ax = sns.barplot(x=top10.index, y=top10.values, hue=top10.index, palette="pastel")
for container in ax.containers:
    ax.bar_label(container, fmt="%d")
ax.ticklabel_format(style='plain', axis='y')
plt.title("Los 10 paises con la mayor cantidad de lanzamientos")
plt.tight_layout()
plt.savefig(os.path.join(dir_actual, "top10_paises_lanzamientos.jpg"))
plt.close()

#boxplot
plt.figure(figsize=(10, 7))
sns.boxplot(data=sample, x='format', y='year', hue='format', palette='pastel')
plt.xticks(rotation=90)
plt.title('Outliers y dispersión de formatos a través de los años')
plt.tight_layout()
plt.savefig(os.path.join(dir_actual, "boxplot_formats.jpg"))
plt.close()

#circular
formats_per_year = df['format'].value_counts()
top5 = formats_per_year.head(5)
otros = formats_per_year.iloc[5:].sum()
final = pd.concat([top5, pd.Series({'Otros': otros})])
plt.figure(figsize=(8,8))
plt.pie(final, labels=final.index, autopct='%1.1f%%', startangle=90)
plt.title('Los 5 formatos con mayores lanzamientos a través de los años')
plt.tight_layout()
plt.savefig(os.path.join(dir_actual, "top5_formats_pie.jpg"))
plt.close()
#histogramas
plt.figure(figsize=(12,5))
sns.histplot(data=df, x='year', hue='genre', palette='pastel',multiple='stack', bins=30)
plt.title('Lanzamientos por año, por género')
plt.tight_layout()
plt.savefig(os.path.join(dir_actual, "topgenres_peryear_histogram.jpg"))
plt.close()