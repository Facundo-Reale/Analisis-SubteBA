import os 
import pandas as pd 

folder_path = r'D:\Usuario\Documents\Cosas python\Prediccion subte' 
dataframes = [] 

for filename in os.listdir(folder_path): 
    filepath = os.path.join(folder_path, filename) 
    if not filename.endswith('.csv') or not os.path.isfile(filepath): 
        continue 

    try: 
        df_raw = pd.read_csv(filepath, encoding='utf-8-sig', header=None, quoting=3) 
    except UnicodeDecodeError: 
        print(f"⚠️ {filename} no es utf-8. Probando latin1...") 
        df_raw = pd.read_csv(filepath, encoding='latin1', header=None, quoting=3) 

    df_split = ( 
        df_raw[0] 
        .str.replace('"', '')    
        .str.split(';', expand=True)) 
    df_split.columns = df_split.iloc[0] 
    df_split = df_split[1:].reset_index(drop=True) 
    required_cols = ['FECHA', 'ESTACION', 'LINEA', 'pax_TOTAL'] 
    if not all(col in df_split.columns for col in required_cols): 
        print(f"⚠️ {filename} falta columnas: {[col for col in required_cols if col not in df_split.columns]}") 
        continue 

    df = df_split[required_cols].copy() 
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d/%m/%Y', errors='coerce') 
    df['ESTACION'] = df['ESTACION'].astype('category') 
    df['LINEA'] = df['LINEA'].astype('category') 
    df['pax_TOTAL'] = pd.to_numeric(df['pax_TOTAL'], errors='coerce') 

    dataframes.append(df) 
    print(f"✅ {filename} procesado y agregado.") 

# Concatenar y guardar 
df_unido = pd.concat(dataframes, ignore_index=True) 
print(f"\n🎯 DataFrame unido: {df_unido.shape}") 

df_unido.to_parquet('df_unido_final.parquet', index=False) 
print("✅ Archivo guardado como df_unido_final.parquet")