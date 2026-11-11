import matplotlib.pyplot as plt
import os
import pandas as pd
import kagglehub
from wordcloud import WordCloud
from pathlib import Path

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
# obtener los formatos con mayor peso dentro de discogs
formatos = df['format'].value_counts()
top30 = formatos.head(30).to_dict()

wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='inferno',
                      max_words=30).generate_from_frequencies(top30)
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Nube de palabras basada en los top 30 formatos en Discogs')
plt.show()
wordcloud.to_file(f'{os.path.dirname(__file__)}/wordcloud_30_formatos.png')