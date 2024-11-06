import pandas as pd
import nltk
import skfuzzy as fuzz
import numpy as np
import time
from nltk.sentiment import SentimentIntensityAnalyzer
positivos = 0
negativos = 0
neutrales = 0
hora_inicio = time.time()
nltk.download('vader_lexicon') # Descargar el léxico de VADER

df = pd.read_csv('test_data_procesado.csv')
sia = SentimentIntensityAnalyzer()

# 2. Lexicón de sentimientos difusos
# Función para calcular el sentimiento difuso
def puntuacion_sentimiento_difuso(text):
    scores = sia.polarity_scores(text)
    puntuacion_pos = scores['pos'] # Calcula el puntaje positivo del tweet
    puntuacion_neg = scores['neg'] # Calcula el puntaje negativo del tweet
    return puntuacion_pos, puntuacion_neg # Devuelve los puntajes positivo y negativo

# Aplicar la función a cada oración en el DataFrame
df[['puntuacion_pos', 'puntuacion_neg']] = df['sentence'].apply(puntuacion_sentimiento_difuso).apply(pd.Series)

minimoPos = df['puntuacion_pos'].min()
maximoPos = df['puntuacion_pos'].max()
medioPos = (minimoPos + maximoPos) / 2
minimoNeg = df['puntuacion_neg'].min()
maximoNeg = df['puntuacion_neg'].max()
medioNeg = (minimoNeg + maximoNeg) / 2


# 3. Fuzzificación
def aplicar_analisis_fuzzy(df):
    global positivos, negativos, neutrales  
    resultados = []
    tiempo_positivos, tiempo_negativos, tiempo_neutrales = [], [], []
    # Definir el universo del discurso y los conjuntos difusos
    x_pos = np.arange(0, 1, 0.1) # Vector de 0 a 1 con paso de 0.1
    x_neg = np.arange(0, 1, 0.1) # Vector de 0 a 1 con paso de 0.1
    x_op = np.arange(0, 10, 1) # Vector de 0 a 10 con paso de 1
    
    # Define las funciones de pertencia en los rangos establecidos (los del universo del discurso)
    pos_bajo = fuzz.trimf(x_pos, [minimoPos, minimoPos, medioPos])
    pos_medio = fuzz.trimf(x_pos, [minimoPos, medioPos, maximoPos])
    pos_alto = fuzz.trimf(x_pos, [medioPos, maximoPos, maximoPos])
    neg_bajo = fuzz.trimf(x_neg, [minimoNeg, minimoNeg, medioNeg])
    neg_medio = fuzz.trimf(x_neg, [minimoNeg, medioNeg, maximoNeg])
    neg_alto = fuzz.trimf(x_neg, [medioNeg, maximoNeg, maximoNeg])
    op_neg = fuzz.trimf(x_op, [0, 0, 5])
    op_neu = fuzz.trimf(x_op, [0, 5, 10])
    op_pos = fuzz.trimf(x_op, [5, 10, 10])

    
    
    for i, row in df.iterrows():
        hora_inicio = time.time()
        
        puntuacion_pos, puntuacion_neg = puntuacion_sentimiento_difuso(row['sentence'])

        # Fuzzificación
        pos_bajo_val = fuzz.interp_membership(x_pos, pos_bajo, puntuacion_pos) # Calcula la membresía del puntaje positivo en el conjunto 'bajo'
        pos_medio_val = fuzz.interp_membership(x_pos, pos_medio, puntuacion_pos) # Calcula la membresía del puntaje positivo en el conjunto 'medio'
        pos_alto_val = fuzz.interp_membership(x_pos, pos_alto, puntuacion_pos) # Calcula la membresía del puntaje positivo en el conjunto 'alto'

        neg_bajo_val = fuzz.interp_membership(x_neg, neg_bajo, puntuacion_neg) # Calcula la membresía del puntaje negativo en el conjunto 'bajo'
        neg_medio_val = fuzz.interp_membership(x_neg, neg_medio, puntuacion_neg) # Calcula la membresía del puntaje negativo en el conjunto 'medio'
        neg_alto_val = fuzz.interp_membership(x_neg, neg_alto, puntuacion_neg) # Calcula la membresía del puntaje negativo en el conjunto 'alto'

        # Reglas manuales para calcular sentimiento final
        w_r1 = min(pos_bajo_val, neg_bajo_val)
        w_r2 = min(pos_medio_val, neg_bajo_val)
        w_r3 = min(pos_alto_val, neg_bajo_val)
        w_r4 = min(pos_bajo_val, neg_medio_val)
        w_r5 = min(pos_medio_val, neg_medio_val)
        w_r6 = min(pos_alto_val, neg_medio_val)
        w_r7 = min(pos_bajo_val, neg_alto_val)
        w_r8 = min(pos_medio_val, neg_alto_val)
        w_r9 = min(pos_alto_val, neg_alto_val)

        # Agregación de reglas
        w_neg = max(w_r4, w_r7, w_r8)
        w_pos = max(w_r2, w_r3, w_r6)
        w_neu = max(w_r1, w_r5, w_r9)
        
        op_activacion_bajo = np.fmin(w_neg, op_neg)
        op_activacion_medio = np.fmin(w_neu, op_neu)
        op_activacion_alto = np.fmin(w_pos, op_pos)

        # Verifica cuál es el valor máximo de los tres conjuntos de salida
        agregado = np.fmax(op_activacion_bajo, op_activacion_medio, op_activacion_alto)

        # Defuzzificación (Centroide)
        # Retorna el centro del área bajo la curva de la función de membresía agregada
        coa = fuzz.centroid(x_op, agregado)

        # Clasificación y tiempo
        output = ''

        if 0 < coa < 3.3:
            output = 'Negativo'
            negativos += 1
        elif 3.3 < coa < 6.7:
            output = 'Neutral'
            neutrales += 1
        else:
            output = 'Positivo'
            positivos += 1

        puntuacion_sentimiento = coa
        tiempo_ejecucion = time.time() - hora_inicio
        label_sentimiento = output

        if label_sentimiento == "Positivo":
            tiempo_positivos.append(tiempo_ejecucion)
        elif label_sentimiento == "Negativo":
            tiempo_negativos.append(tiempo_ejecucion)
        else:
            tiempo_neutrales.append(tiempo_ejecucion)

        resultados.append([
            row['sentence'], row['sentiment'], puntuacion_pos, puntuacion_neg,
            puntuacion_sentimiento, tiempo_ejecucion, label_sentimiento
        ])

    # Crea un dataframe con los resultados
    resultado_df = pd.DataFrame(resultados, columns=[
        'Oración original', 'label original', 'puntaje_positivo', 'puntaje_negativo',
        'Resultado de inferencia', 'tiempo de ejecucion', "Label de inferencia"
    ])

    return resultado_df

df_resultado = aplicar_analisis_fuzzy(df)
df_resultado.to_csv('dataset_sentimientos_fuzzy_con_inferencia.csv', sep=';', index=False)
hora_final = time.time()
tiempo_de_ejecucion_total = hora_final - hora_inicio
print(f"Tiempo de ejecución promedio total: {df_resultado['tiempo de ejecucion'].mean()} segundos")
print(f"Tiempo total de ejecución: {tiempo_de_ejecucion_total} segundos")
print(f"Total positivos: {positivos}")
print(f"Total negativos: {negativos}")
print(f"Total neutrales: {neutrales}")
