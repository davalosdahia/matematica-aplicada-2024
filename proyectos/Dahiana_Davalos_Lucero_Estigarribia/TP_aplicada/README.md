# TP Aplicada: Análisis de Sentimientos con Fuzzy Logic y NLTK

Este proyecto aplica lógica difusa y análisis de sentimientos utilizando NLTK y `scikit-fuzzy` para procesar textos y clasificarlos en positivo, negativo o neutral.

## Instalación y configuración

1. Clonar este repositorio:
    ```bash
    git clone https://github.com/davalosdahia/matematica-aplicada-2024.git
    cd proyectos/Dahiana_Davalos_Lucero_Estigarribia/TP_aplicada
    ```

2. Crear y activar un entorno virtual:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # En Windows
    source venv/bin/activate  # En Linux/Mac
    ```

3. Instalar dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Ejecución de scripts

Generar el dataset y preparar el léxico:
```bash
python createDataset.py
python lexicon.py
```
