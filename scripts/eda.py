import warnings
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import joypy
import matplotlib.cm as cm
import matplotlib.image as mpimg 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox 
import os
from PIL import Image 
from matplotlib.ticker import FuncFormatter
import re
from thefuzz import process
import folium
from geopy.distance import geodesic
from branca.element import Template, MacroElement
warnings.filterwarnings("ignore")

#Cargar datos
df_unido = pd.read_parquet(r'D:\Usuario\Documents\Cosas python\Prediccion subte\Estructura GitHub\data\datos_subte.parquet')

#Transformar con polars para agrupación rápida
df_pl = pl.from_pandas(df_unido)

resultado = (
    df_pl
    .group_by(["FECHA", "ESTACION", "LINEA"])
    .agg(pl.sum("pax_TOTAL").alias("PAX_TOTAL")))

df_agg = resultado.to_pandas()

#Tipos y ordenamiento
df_agg['FECHA'] = pd.to_datetime(df_agg['FECHA'], errors='coerce')
df_agg['ESTACION'] = df_agg['ESTACION'].astype('category')
df_agg['LINEA'] = df_agg['LINEA'].astype('category')
df_agg['PAX_TOTAL'] = pd.to_numeric(df_agg['PAX_TOTAL'], errors='coerce')
df_agg = df_agg.sort_values('FECHA').reset_index(drop=True)

#Variables calendario
df_agg['DIA_SEMANA'] = df_agg['FECHA'].dt.day_name()
df_agg['MES'] = df_agg['FECHA'].dt.month
df_agg['AÑO'] = df_agg['FECHA'].dt.year

pax_diario_linea = (
    df_agg
    .groupby(['LINEA', 'FECHA'])['PAX_TOTAL']
    .sum()
    .reset_index(name='PAX_DIARIO_TOTAL'))

promedio_diario_linea = (
    pax_diario_linea
    .groupby('LINEA')['PAX_DIARIO_TOTAL']
    .mean()
    .reset_index(name='PAX_PROMEDIO')
    .sort_values(by='PAX_PROMEDIO', ascending=False))
df_promedio =promedio_diario_linea.reset_index()
df_promedio =df_promedio.rename(columns= {"PAX_TOTAL":"PAX_PROMEDIO"})
df_promedio = df_promedio.sort_values(by='PAX_PROMEDIO', ascending=False)


colores_lineas = {
    'LineaA': '#00BFFF',
    'LineaB': '#FF0000',
    'LineaC': '#00008B',
    'LineaD': '#228B22',
    'LineaE': '#800080',
    'LineaH': '#DAA520'}

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'axes.titlesize': 28,
    'axes.titleweight': 'bold',
    'axes.labelsize': 20,
    'axes.labelcolor': 'dimgray',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.edgecolor': 'gray',
    'grid.color': 'lightgray',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.8,
    'figure.titlesize': 28,
    'figure.titleweight': 'bold',
    'axes.labelweight': 'bold'})

# --- Gráfico 1: Pasajeros promedio por línea (Barplot) ---
fig, ax = plt.subplots(figsize=(12,7))
sns.barplot(data=df_promedio,
            x='PAX_PROMEDIO', y='LINEA',
            palette=colores_lineas,
            ax=ax,
            order=df_promedio['LINEA'])

posicion_texto_x = 500
ax.set_yticklabels([])
for i, bar in enumerate(ax.patches):
    y = bar.get_y() + bar.get_height() / 2
    ancho = bar.get_width()
    ax.text(ancho - 16800, y,
            f'{ancho:,.0f}'.replace(',', '.'),
            va='center', ha='left',
            color='white',
            fontsize=22, 
            fontweight='bold')
    nombre_linea = df_promedio.iloc[i]['LINEA']
   
    ax.text(posicion_texto_x, y, nombre_linea,
            va='center',ha='left',
            color='white',
            fontsize=22,
            fontweight='bold')

plt.title(
    "Promedio diario de pasajeros por línea (2022-2024)", 
    fontsize=24,
    fontweight='bold',
    color='#333333',
    pad=10)

plt.xlabel("Pasajeros promedio diarios", fontsize=20, fontweight='bold', color='#333333')
plt.ylabel("Línea", fontsize=20, fontweight='bold', color='#333333')
plt.tight_layout()
plt.grid(False)
plt.show()


# --- Gráfico 2: Tendencia Mensual del Promedio Diario de Pasajeros por Línea (Lineplot) ---
df_tendencia_linea = df_agg.groupby(['FECHA', 'LINEA'])['PAX_TOTAL'].sum().reset_index()
df_tendencia_mensual = df_tendencia_linea.copy()
df_tendencia_mensual['MES_AÑO'] = df_tendencia_mensual['FECHA'].dt.to_period('M')

df_tendencia_mensual_agrupado = df_tendencia_mensual.groupby(['MES_AÑO', 'LINEA'])['PAX_TOTAL'].mean().reset_index()
df_tendencia_mensual_agrupado['MES_AÑO'] = df_tendencia_mensual_agrupado['MES_AÑO'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(20, 12))
sns.lineplot(
    data=df_tendencia_mensual_agrupado,
    x='MES_AÑO',
    y='PAX_TOTAL',
    hue='LINEA',
    palette=colores_lineas,
    linewidth=3.5,
    ax=ax
)

last_points = df_tendencia_mensual_agrupado.loc[df_tendencia_mensual_agrupado.groupby('LINEA')['MES_AÑO'].idxmax()]
ruta_imagenes = r'D:\Usuario\Documents\Cosas python\Prediccion subte\Estructura GitHub\data\\'
target_image_width = 28 
target_image_height = 28
ax.legend(title='LÍNEA',
          fontsize='x-large',
          title_fontsize='xx-large',
          bbox_to_anchor=(0.99, 1), 
          loc='upper left',
          borderaxespad=1,
          frameon=False 
         )
vertical_offsets = {
    'LineaA': 6,
    'LineaB': 0,
    'LineaC': 0,
    'LineaD': 0,
    'LineaE': 0,
    'LineaH': 8
}
for index, row in last_points.iterrows():
    linea = row['LINEA']
    fecha = row['MES_AÑO']
    pax_total = row['PAX_TOTAL']

    image_path = f"{ruta_imagenes}{linea}.png"

    if os.path.exists(image_path):
        img_pil = Image.open(image_path)
        img_pil = img_pil.resize((target_image_width, target_image_height), Image.LANCZOS)
        img_mpl = np.asarray(img_pil)

        imagebox = OffsetImage(img_mpl, zoom=1)


        ab = AnnotationBbox(imagebox, (fecha, pax_total),
                            xycoords='data',
                            boxcoords="offset points",
                            xybox=(8, vertical_offsets.get(linea, 0)), 
                            frameon=False)
        ax.add_artist(ab)
    else:
        print(f"Advertencia: No se encontró la imagen para {linea} en {image_path}")
        ax.annotate(linea, (fecha, pax_total),
                    xytext=(10, 0),
                    textcoords='offset points',
                    fontsize=18,
                    color=colores_lineas.get(linea),
                    ha='left', va='center',
                    fontweight='bold')


plt.xlabel("Fecha",
           fontsize=24,
           color='#333333',
           fontweight='bold')

plt.ylabel("Promedio Diario de Pasajeros",
           fontsize=24,
           color='#333333',
           fontweight='bold')


plt.xticks(rotation=45,
           ha='right',
           fontsize=18,
           color='#555555')

plt.yticks(fontsize=18,
           color='#555555') 

formatter = mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', '.'))
plt.gca().yaxis.set_major_formatter(formatter)

plt.grid(False)
fig.suptitle(
    "Tendencia Mensual del Promedio Diario de Pasajeros por Línea (2022-2024)",
    fontsize=32,
    fontweight='bold',
    color='#333333',
    y=0.95 
)


fig.subplots_adjust(top=0.9,
                    right=0.92,
                    left=0.12,
                    bottom=0.18)

plt.show()

# --- Gráfico 3: Distribución del Total Diario de Pasajeros en el Sistema (Histograma) ---
df_daily_total = df_agg.groupby('FECHA')['PAX_TOTAL'].sum().reset_index()
df_daily_total = df_daily_total.rename(columns={'PAX_TOTAL': 'PAX_TOTAL_DIARIO_SISTEMA'})

df_daily_total['DIA_SEMANA'] = df_daily_total['FECHA'].dt.day_name()
orden_dias_espanol = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
df_daily_total['DIA_SEMANA'] = pd.Categorical(df_daily_total['DIA_SEMANA'],
                                            categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                                            ordered=True)
df_daily_total['DIA_SEMANA'] = df_daily_total['DIA_SEMANA'].cat.rename_categories({
    "Monday":"Lunes",
    "Tuesday":"Martes",
    "Wednesday":"Miércoles",
    "Thursday":"Jueves",
    "Friday":"Viernes",
    "Saturday":"Sábado",  
    "Sunday":"Domingo"})

df_daily_total_sorted = df_daily_total.sort_values('DIA_SEMANA')



plt.figure(figsize=(12, 7))
ax = sns.histplot(
    df_daily_total['PAX_TOTAL_DIARIO_SISTEMA'],
    bins=50,
    kde=True,
    color='#00BFFF',
    edgecolor='black',
    alpha=0.7)


plt.title(
    'Distribución del Total Diario de Pasajeros en el Sistema',
    loc='center',
    fontsize=24,
    fontweight='bold',
    color='dimgray')


plt.xlabel(
    'Total de Pasajeros Diarios en el Sistema',
    fontsize=18,
    color='dimgray',
    fontweight='bold')

plt.ylabel(
    'Frecuencia de Días',
    fontsize=18,
    color='dimgray',
    fontweight='bold')



def miles_con_punto(x, _):
    return f'{x:,.0f}'.replace(',', '.')
formateador = FuncFormatter(miles_con_punto)

ax.xaxis.set_major_formatter(formateador)
plt.tight_layout()
plt.grid(False)
plt.show()

# --- Gráfico 4: Distribución de Pasajeros por Día de la Semana (Joyplot) ---
fig, axes = joypy.joyplot(
    df_daily_total_sorted,
    by='DIA_SEMANA',
    column='PAX_TOTAL_DIARIO_SISTEMA',
    figsize=(12, 8),
    overlap=0,
    grid=False,
    title=None,
    xlabels=True,
    ylabels=True,
    fade=True,
    fill=True,
    linecolor='black',
    linewidth=0.1,
    colormap=cm.Paired,
    ylim='own',
    x_range=[0, df_daily_total['PAX_TOTAL_DIARIO_SISTEMA'].max() * 1.1])

plt.xlabel("Total de Pasajeros Diarios en el Sistema", fontsize=20, color='dimgray', fontweight='bold') 
plt.ylabel("Día de la Semana", fontsize=20, color='dimgray', fontweight='bold')
plt.title(
    "Distribución del Total Diario de Pasajeros en el Sistema por Día de la Semana",
    fontsize=28,
    color='dimgray',
    fontweight='bold',
    pad=20)

ax_list = fig.get_axes()
for ax_single in ax_list:
    ax_single.xaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False, useMathText=False))
    ax_single.xaxis.get_major_formatter().set_scientific(False)
    ax_single.xaxis.set_major_formatter(FuncFormatter(miles_con_punto))
    ax_single.tick_params(axis='x', labelsize=16) 
    ax_single.tick_params(axis='y', labelsize=16) 

plt.tight_layout()
plt.show()

rango_pasajeros_por_dia_sistema = df_daily_total.groupby('DIA_SEMANA')['PAX_TOTAL_DIARIO_SISTEMA'].agg(['min', 'max']).reset_index()

rango_pasajeros_por_dia_sistema['min'] = rango_pasajeros_por_dia_sistema['min'].apply(lambda x: f'{x:,.0f}')
rango_pasajeros_por_dia_sistema['max'] = rango_pasajeros_por_dia_sistema['max'].apply(lambda x: f'{x:,.0f}')

print(rango_pasajeros_por_dia_sistema)

df_daily_total['MES'] = df_daily_total['FECHA'].dt.month
meses_espanol = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}
df_daily_total['MES_NOMBRE'] = df_daily_total['MES'].map(meses_espanol)

# Clasificar tipo de día
df_daily_total['TIPO_DIA'] = df_daily_total['DIA_SEMANA'].isin(['Sábado', 'Domingo']).map({True: 'Fin de Semana', False: 'Día de Semana'})

plt.figure(figsize=(16,9))
palette_tipo_dia = {
    'Día de Semana': '#00BFFF',  # celeste
    'Fin de Semana': '#FF6347'   # rojo tomate
}

ax = sns.boxplot(
    data=df_daily_total,
    x='MES_NOMBRE',
    y='PAX_TOTAL_DIARIO_SISTEMA',
    hue='TIPO_DIA',
    palette=palette_tipo_dia,
    width=0.6,
    fliersize=8,
    linewidth=1.2
)

plt.title(
    "Distribución del Total Diario de Pasajeros por Mes y Tipo de Día",
    fontsize=28,
    fontweight='bold',
    color='dimgray',
    pad=20
)
plt.xlabel("Mes", fontsize=22, fontweight='bold', color='dimgray')
plt.ylabel("Pasajeros Diarios", fontsize=22, fontweight='bold', color='dimgray')

plt.xticks(rotation=45, fontsize=16, color='dimgray')
plt.yticks(fontsize=16, color='dimgray')

ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'.replace(',', '.')))

plt.grid(False)
plt.legend(title='Tipo de Día', title_fontsize='x-large', fontsize=16, loc='upper left', frameon=False)
plt.tight_layout()
plt.show()
# Carga cordenadas
coordenadas = pd.read_csv(r"D:\Usuario\Documents\Cosas python\Prediccion subte\Estructura GitHub\data\estaciones-de-subte-geograficas.csv", sep=";")
def limpiar_y_normalizar_nombre_estacion(nombre):
    if pd.isna(nombre): return nombre
    nombre_original_limpio = str(nombre).upper().strip()

    mojibake_map = {
        'Ã\x81': 'Á', 'Ã\x89': 'É', 'Ã\x8d': 'Í', 'Ã\x93': 'Ó', 'Ã\x9a': 'Ú',
        'Ã\x91': 'Ñ', 'Ã\x9c': 'Ü', 'Ã‘': 'Ñ', 'Ã‰': 'É', 'Ã“': 'Ó', 'Ãš': 'Ú',
        'Ãœ': 'Ü', 'Â°': '°', '±': 'Ñ',
        'AGÜERO': 'AGUERO', 'ECHEVERRIA': 'ECHEVERRÍA', 'CORDOBA': 'CÓRDOBA'
    }
    for k, v in mojibake_map.items():
        nombre_original_limpio = nombre_original_limpio.replace(k, v)

    nombre_limpio_temp = re.sub(r'[^A-Z0-9ÁÉÍÓÚÑÜ°\s]', ' ', nombre_original_limpio)
    nombre_limpio_temp = re.sub(r'\s+', ' ', nombre_limpio_temp).strip()

    mapeo_nombres_a_estandares = {
        "PATRICIOS": "PARQUE PATRICIOS", "RETIRO E": "RETIRO",
        "SANTA FE": "SANTA FE - CARLOS JAUREGUI", "PASTEUR": "PASTEUR - AMIA",
        "PZA DE LOS VIRREYES": "PLAZA DE LOS VIRREYES - EVA PERON",
        "GENERAL SAN MARTIN": "SAN MARTIN", "MEDRANO": "ALMAGRO - MEDRANO",
        "HUMBERTO I": "HUMBERTO 1°", "FLORES": "SAN JOSÉ DE FLORES",
        "CONGRESO": "CONGRESO - PDTE DR RAÚL R ALFONSÍN",
        "INCLAN": "INCLAN - MEZQUITA AL AHMAD",
        "MALABIA": "MALABIA - OSVALDO PUGLIESE",
        "LOS INCAS": "DE LOS INCAS -PQUE CHAS", "SCALABRINI ORTIZ": "R.SCALABRINI ORTIZ",
        "GENERAL BELGRANO": "BELGRANO", "AVENIDA DE MAYO": "AV. DE MAYO",
        "CARLOS PELLEGRINI": "C. PELLEGRINI",
        "MINISTRO CARRANZA": "MINISTRO CARRANZA - MIGUEL ABUELO",
        "ROSAS": "JUAN MANUEL DE ROSAS - VILLA URQUIZA",
        "ONCE": "ONCE - 30 DE DICIEMBRE",
        "TRIBUNALES": "TRIBUNALES - TEATRO COLÓN",
        "TRONADOR": "TRONADOR - VILLA ORTÚZAR",
        "AVENIDA LA PLATA": "AV. LA PLATA", "ENTRE RIOS": "ENTRE RIOS - RODOLFO WALSH",
        "PLAZA MISERERE": "PLAZA DE MISERERE",
        "FACULTAD DE DERECHO": "FACULTAD DE DERECHO - JULIETA LANTERI",
        "MARIANO MORENO": "MORENO",
        "PUEYRREDON D": "PUEYRREDON", "CALLAO B": "CALLAO - MAESTRO ALFREDO BRAVO",
        "CALLAO D": "CALLAO", "INDEPENDENCIA H": "INDEPENDENCIA",
        "INDEPENDENCIA C": "INDEPENDENCIA", "INDEPENDENCIA E": "INDEPENDENCIA",
        "RETIRO C": "RETIRO", "RETIRO E": "RETIRO",
        "PUEYRREDON B": "PUEYRREDON"
    }
    return mapeo_nombres_a_estandares.get(nombre_limpio_temp, nombre_limpio_temp)


coordenadas["ESTACION_LIMPIA"] = coordenadas["estacion"].apply(limpiar_y_normalizar_nombre_estacion)
coordenadas["LINEA_LIMPIA"] = coordenadas["linea"].astype(str).str.upper().str.strip()
coordenadas["CLAVE_MERGE"] = coordenadas["ESTACION_LIMPIA"] + "_" + coordenadas["LINEA_LIMPIA"]
coordenadas_para_merge = coordenadas[['CLAVE_MERGE', 'lat', 'long']].drop_duplicates(subset=['CLAVE_MERGE'])

df_agg["ESTACION_LIMPIA"] = df_agg["ESTACION"].apply(limpiar_y_normalizar_nombre_estacion)
df_agg["LINEA_LIMPIA"] = df_agg["LINEA"].astype(str).str.upper().str[-1]
df_agg["CLAVE_MERGE"] = df_agg["ESTACION_LIMPIA"] + "_" + df_agg["LINEA_LIMPIA"]


df_final = pd.merge(df_agg, coordenadas_para_merge, on='CLAVE_MERGE', how='left')


estaciones_referencia_coords = coordenadas_para_merge['CLAVE_MERGE'].unique().tolist()

def aplicar_fuzzy_matching_a_fila(row, estaciones_referencia, umbral=85):
    if pd.isna(row['lat']):
        nombre_a_buscar = row['CLAVE_MERGE']
        if pd.isna(nombre_a_buscar): return None, 0
        best_match_result = process.extractOne(nombre_a_buscar, estaciones_referencia)
        if best_match_result and best_match_result[1] >= umbral:
            return best_match_result[0], best_match_result[1]
    return row['CLAVE_MERGE'], 100

try:
    df_final[['CLAVE_MERGE_FUZZY_SUG', 'SCORE_FUZZY']] = df_final.apply(
        lambda row: aplicar_fuzzy_matching_a_fila(row, estaciones_referencia_coords),
        axis=1, result_type='expand'
    )

    df_final_con_fuzzy = pd.merge(
        df_final.drop(columns=['lat', 'long'], errors='ignore'),
        coordenadas_para_merge,
        left_on='CLAVE_MERGE_FUZZY_SUG',
        right_on='CLAVE_MERGE',
        how='left',
        suffixes=('_dfagg', '_coords_fuzzy')
    ).rename(columns={'CLAVE_MERGE_coords_fuzzy': 'CLAVE_MERGE_COORDENADAS'})

except ImportError:
    print("The 'thefuzz' library is not installed. To use fuzzy matching, run: pip install thefuzz python-Levenshtein")
    # Fallback if fuzzy matching is not possible
    df_final_con_fuzzy = df_final.copy()
    df_final_con_fuzzy['CLAVE_MERGE_FUZZY_SUG'] = df_final_con_fuzzy['CLAVE_MERGE']
    df_final_con_fuzzy['SCORE_FUZZY'] = 100


df_final_con_fuzzy['ESTACION_FINAL_NORMALIZADA'] = df_final_con_fuzzy['CLAVE_MERGE_FUZZY_SUG'].apply(
    lambda x: x.rsplit('_', 1)[0] if pd.notna(x) and '_' in x else x
)

df_estaciones_coords = df_final_con_fuzzy[[
    'lat',
    'long',
    'ESTACION_LIMPIA',
    'LINEA',
    'FECHA',
    'PAX_TOTAL'
]].copy()

daily_flow = df_estaciones_coords.groupby(['lat', 'long', 'FECHA'])['PAX_TOTAL'].sum().reset_index()
avg_daily_flow = daily_flow.groupby(['lat', 'long'])['PAX_TOTAL'].mean().reset_index()
avg_daily_flow = avg_daily_flow.rename(columns={'PAX_TOTAL': 'PAX_PROMEDIO_DIARIO'})
station_names = df_estaciones_coords.groupby(['lat', 'long'])['ESTACION_LIMPIA'] \
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]) \
                .reset_index()
lineas = df_estaciones_coords.groupby(['lat', 'long'])['LINEA'] \
    .agg(lambda x: x.iloc[0]) \
    .reset_index()

# Paso 5: unir todo
flujo_diario_estaciones = avg_daily_flow \
    .merge(station_names, on=['lat', 'long'], how='left') \
    .merge(lineas, on=['lat', 'long'], how='left')

# Resultado final
print(flujo_diario_estaciones.head())
flujo_diario_estaciones['lat'] = flujo_diario_estaciones['lat'].astype(str).str.replace(',', '.')
flujo_diario_estaciones['long'] = flujo_diario_estaciones['long'].astype(str).str.replace(',', '.')

# Ahora convertir a float
flujo_diario_estaciones['lat'] = pd.to_numeric(flujo_diario_estaciones['lat'], errors='coerce')
flujo_diario_estaciones['long'] = pd.to_numeric(flujo_diario_estaciones['long'], errors='coerce')

# CREACION DE MAPA INTERACTIVO

colores_lineas = {
    'LineaA': '#00BFFF',
    'LineaB': '#FF0000',
    'LineaC': '#00008B',
    'LineaD': '#228B22',
    'LineaE': '#800080',
    'LineaH': '#DAA520'
}

def ordenar_por_cercania(puntos):
    if len(puntos) <= 1:
        return puntos
    puntos = puntos.copy()
    orden = [puntos.pop(0)]
    while puntos:
        ultimo = orden[-1]
        siguiente = min(puntos, key=lambda p: geodesic(ultimo, p).meters)
        orden.append(siguiente)
        puntos.remove(siguiente)
    return orden

def ordenar_desde_origen(puntos, origen):
    if len(puntos) <= 1:
        return puntos
    puntos = puntos.copy()
    orden = [origen]
    puntos.remove(origen)
    while puntos:
        siguiente = min(puntos, key=lambda p: geodesic(orden[-1], p).meters)
        orden.append(siguiente)
        puntos.remove(siguiente)
    return orden

lat_centro = flujo_diario_estaciones['lat'].mean()
lon_centro = flujo_diario_estaciones['long'].mean()

mapa = folium.Map(location=[lat_centro, lon_centro], zoom_start=12, tiles='CartoDB positron')
max_pax = flujo_diario_estaciones['PAX_PROMEDIO_DIARIO'].max()

for _, row in flujo_diario_estaciones.iterrows():
    radio = 3 + 12 * (row['PAX_PROMEDIO_DIARIO'] / max_pax)
    color = colores_lineas.get(row['LINEA'], '#000000')

    popup_html = (
        f"<b>Estación:</b> {row['ESTACION_LIMPIA']}<br>"
        f"<b>Línea:</b> {row['LINEA']}<br>"
        f"<b>Flujo promedio diario:</b> {row['PAX_PROMEDIO_DIARIO']:.0f}"
    )
    tooltip_text = f"{row['ESTACION_LIMPIA']} ({row['LINEA']}) - {row['PAX_PROMEDIO_DIARIO']:.0f} pax/día"

    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=radio,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=tooltip_text
    ).add_to(mapa)

for linea, color in colores_lineas.items():
    subdf = flujo_diario_estaciones[flujo_diario_estaciones['LINEA'] == linea].dropna(subset=['lat', 'long'])
    puntos = list(zip(subdf['lat'], subdf['long']))

    if linea == 'LineaA':
        origen_df = subdf[subdf['ESTACION_LIMPIA'].str.contains('plaza de mayo', case=False)]
        if not origen_df.empty:
            origen = (origen_df.iloc[0]['lat'], origen_df.iloc[0]['long'])
            puntos_ordenados = ordenar_desde_origen(puntos, origen)
        else:
            puntos_ordenados = ordenar_por_cercania(puntos)
    else:
        puntos_ordenados = ordenar_por_cercania(puntos)

    folium.PolyLine(
        locations=puntos_ordenados,
        color=color,
        weight=4,
        opacity=0.7,
        tooltip=f'Línea {linea}'
    ).add_to(mapa)

legend_html = """
{% macro html(this, kwargs) %}
<div style="
    position: fixed; 
    bottom: 50px; left: 50px; width: 180px; height: 170px; 
    background-color: white; 
    border: 2px solid grey; 
    z-index: 9999; 
    font-size: 14px;
    padding: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
">
    <b>Líneas de Subte</b><br>
    &nbsp;<i style="background:#00BFFF;width:18px;height:18px;float:left;margin-right:8px;"></i>Linea A<br>
    &nbsp;<i style="background:#FF0000;width:18px;height:18px;float:left;margin-right:8px;"></i>Linea B<br>
    &nbsp;<i style="background:#00008B;width:18px;height:18px;float:left;margin-right:8px;"></i>Linea C<br>
    &nbsp;<i style="background:#228B22;width:18px;height:18px;float:left;margin-right:8px;"></i>Linea D<br>
    &nbsp;<i style="background:#800080;width:18px;height:18px;float:left;margin-right:8px;"></i>Linea E<br>
    &nbsp;<i style="background:#DAA520;width:18px;height:18px;float:left;margin-right:8px;"></i>Linea H<br>
</div>
{% endmacro %}
"""

titulo_html = """
{% macro html(this, kwargs) %}
<div style="
    position: fixed;
    top: 10px; left: 50%;
    transform: translateX(-50%);
    z-index: 9999;
    background-color: white;
    padding: 6px 12px;
    border-radius: 5px;
    font-weight: bold;
    font-size: 20px;
    font-family: Arial, sans-serif;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
">
    Flujo diario promedio de pasajeros por estación de subte
</div>
{% endmacro %}
"""

titulo = MacroElement()
titulo._template = Template(titulo_html)
mapa.get_root().add_child(titulo)

legend = MacroElement()
legend._template = Template(legend_html)
mapa.get_root().add_child(legend)

ruta_salida = r"D:\Usuario\Documents\Cosas python\Prediccion subte\Estructura GitHub\data\flujo_estaciones_final.html"
mapa.save(ruta_salida)
