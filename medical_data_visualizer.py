import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargar los datos
df = pd.read_csv('medical_examination.csv')

# 2. Agregar la columna 'overweight'
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3. Normalizar datos: 0 es bueno, 1 es malo
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Función para dibujar el gráfico categórico
def draw_cat_plot():
    # 5. Crear el DataFrame df_cat usando pd.melt
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. Agrupar y reestructurar datos divididos por cardio con los conteos
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Crear el gráfico catplot usando seaborn
    g = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar'
    )

    # 8. Obtener la figura para la salida
    fig = g.fig

    # 9. Guardar la imagen (No modificar)
    fig.savefig('catplot.png')
    return fig

# 10. Función para dibujar el mapa de calor
def draw_heat_map():
    # 11. Limpiar datos en df_heat según los percentiles y presión arterial
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calcular la matriz de correlación
    corr = df_heat.corr()

    # 13. Generar una máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Configurar la figura de matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Graficar la matriz de correlación con seaborn.heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )

    # 16. Guardar la imagen (No modificar)
    fig.savefig('heatmap.png')
    return fig