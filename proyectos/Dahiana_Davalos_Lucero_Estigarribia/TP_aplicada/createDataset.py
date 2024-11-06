import pandas as pd
from tqdm import tqdm 
import re

# Ruta al archivo del dataset
ruta_archivo = 'test_data.csv'  # Reemplaza con la ruta a tu archivo .csv

# Carga el archivo CSV en un DataFrame de Pandas
test_data = pd.read_csv(ruta_archivo, encoding='ISO-8859-1')


# Función de limpieza
def limpiar_texto(texto: str):

    # Quita URLs (http o https)
    texto = re.sub(r'http[s]?://\S+', '', texto)
    # Quita menciones (@usuario)
    texto = re.sub(r' www\S+', '', texto)
    # Quita hashtags (#hashtag)
    texto = re.sub(r'#\w+', '', texto)
    # Quita guiones medios y bajos (y los convierte en espacios)
    texto = re.sub(r'[-_]+', ' ', texto) 
    # Quita signos de puntuación y números, excepto espacios y letras
    texto = re.sub(r'[^\w\s]', '', texto)
    # Quita múltiple espacios
    texto = re.sub(r'\s+', ' ' , texto)
    texto = re.sub(r'_+', ' ' , texto)
    # Convierte a minúsculas y quita espacios sobrantes
    texto = texto.strip().lower().encode('ascii', 'ignore').decode()
    return texto

# Habilita tqdm para Pandas
tqdm.pandas()

test_data['sentence'] = test_data['sentence'].apply(limpiar_texto)

# Guarda el dataset procesado en un nuevo archivo CSV
test_data.to_csv('test_data_procesado.csv', index=False)