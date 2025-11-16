# Análisis de Ventas con Machine Learning

Aplicación web para analizar datos de ventas, generar visualizaciones y predecir ventas futuras usando modelos de regresión.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
streamlit run app/app.py
```

## Características

- Descarga plantillas CSV/Excel con formato correcto
- Carga y limpia datos automáticamente
- Visualizaciones de ventas por región y categoría
- Entrenamiento de modelos de regresión (Linear/RandomForest)
- Predicciones interactivas

## Estructura

```
├── app/             # Aplicación Streamlit
├── src/             # Pipeline de procesamiento y ML
├── data/            # Datos de ejemplo
├── models/          # Modelos entrenados
└── notebooks/       # Análisis exploratorio
```

## Requisitos mínimos

- Python 3.9+
- Mínimo 10 registros en el dataset para entrenar modelos

