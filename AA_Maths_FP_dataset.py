# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 13:35:14 2025

@author: David
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 12:39:09 2025

@author: David
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler





data = pd.read_csv('C:/Users/david/OneDrive/Documentos/TFG MATEMÁTICAS/FoxInsight.csv')

#Filtro de pacientes CON PARKINSON:
data_PD=data.loc[(data['CurrPDDiag']==1)].copy()
del data, data_PD['CurrPDDiag']


#VARIABLE RESPUESTA:

var_live = [x for x in data_PD.columns if x.startswith("Live")]
data_PD[var_live]=(data_PD[var_live].replace(5, np.nan))
data_PD['CalidadVida']=data_PD[var_live].isin([3, 4]).any(axis=1).map({True: "No aceptable", False: "Aceptable"})

#Se eliminan los registros donde no esté informada:
data_PD = data_PD.loc[~((data_PD["CalidadVida"] == "Aceptable") & data_PD[var_live].isna().any(axis=1))]
data_PD = data_PD.drop(columns=var_live)


#VARIABLES CUANTITATIVAS:
data_PD=data_PD.rename(columns={'age':'Edad', 'HeightInch':'Altura(in)', 'WeightLbs':'Peso(lb)'})


#VARIABLES CATEGÓRICAS

    #Creamos los indicadores artificiales:
    #Lateralidad
var_lat=[x for x in data_PD.columns if x.startswith("Hands")]
data_PD["Lateralidad"] = data_PD[var_lat].apply(
    lambda row: "Diestro" if all(x in [1, 2] for x in row) 
                else ("Zurdo" if all(x in [4, 5] for x in row) 
                      else np.nan), axis=1)
data_PD = data_PD.drop(columns=var_lat)

    #Lateralidad PD
data_PD["LateralidadPD"] = data_PD["SidePDOnset"].apply(lambda x: "Lado izquierdo" if x == 1 
                                                         else ("Lado derecho" if x == 2 else np.nan))




                                                               

del data_PD["SidePDOnset"]
    
    
    #Dicotomizamos las variables categóricas con más de dos tramos, basándonos en los factores de riesgo encontrados en la bibliografía:
data_PD['EstudiosUni']=pd.cut(data_PD['Education'],bins=[0,4,8], labels=['No', 'Sí'],right=True)
data_PD['Pobreza']=pd.cut(data_PD['Income'],bins=[0,2,6], labels=['Sí', 'No'],right=True)
data_PD['Empleo']=pd.cut(data_PD['Employment'],bins=[0,2,4], labels=['Empleado', 'Desempleado/Jubilado'],right=True)

del data_PD["Education"], data_PD["Income"], data_PD["Employment"]

data_PD=data_PD.rename(columns={'Sex':'Sexo'})
data_PD["Sexo"] = data_PD["Sexo"].replace({1: "Masculino", 2: "Femenino"})

    #Preparamos el resto de variablesc categóricas:
data_PD=data_PD.rename(columns={'RaceAA':'Raza', 'Veteran':'Veterano','MedsCurrPD':'MedicamentosPD',
                                'MedPDProcedNone':'OtrosTratPD', 'FamParkinsonHx':'HistFamPD' })
data_PD['Raza']=data_PD['Raza'].replace({0:'Otra',1:'Negra/Afroamericana'})
data_PD["Veterano"] = data_PD["Veterano"].replace({1: "Sí", 0: "No", 3:np.nan})
data_PD["MedicamentosPD"] = data_PD["MedicamentosPD"].replace({1: "Sí", 0: "No", 3:np.nan})
data_PD["OtrosTratPD"] = data_PD["OtrosTratPD"].replace({0: "Sí", 1: "No"})
data_PD["HistFamPD"] = data_PD["HistFamPD"].replace({0: "No/Desconocido", 1: "Sí", 2: "No/Desconocido", 3:np.nan})







#----------------------------------------------------------------------------------------------------------------------------#

#ESTUDIO DE INCOSISTENCIAS PARA VARIABLES QUE HAN DE PERMANECER CONSTANTES:

variables_revision = ['InitPDDiagAge', 'Sexo', 'Raza', 
                      'Lateralidad', 'LateralidadPD','LocCountry']

# Calcular número de valores únicos por paciente y variable
calc_inconsistencias = (
    data_PD[['fox_insight_id'] + variables_revision]
    .melt(id_vars='fox_insight_id', var_name='variable', value_name='valor')
    .groupby(['fox_insight_id', 'variable'])['valor']
    .nunique()
    .reset_index(name='n_distintos')
)

# Filtrar los casos con más de un valor
inconsistencias = calc_inconsistencias[calc_inconsistencias['n_distintos'] > 1]

del calc_inconsistencias

conteo_por_variable = (
    inconsistencias.groupby('variable')
    .size()
    .reset_index(name='n_pacientes_con_inconsistencias')
    .sort_values(by='n_pacientes_con_inconsistencias', ascending=False)
)


print("Detalle de inconsistencias por paciente y variable:")
print(inconsistencias)

print("\nResumen: número de pacientes con inconsistencias por variable:")
print(conteo_por_variable)

ids_inconsistentes = inconsistencias['fox_insight_id'].unique()

# Filtrar registros completos de esos pacientes
registros_inconsistentes = data_PD[data_PD['fox_insight_id'].isin(ids_inconsistentes)]


#Eliminamos estos registros de la muestra:
data_PD_sin_inconsistencias = data_PD[~data_PD['fox_insight_id'].isin(ids_inconsistentes)]


# Para cada variable, propagar el valor por paciente
for var in variables_revision:
    data_PD_sin_inconsistencias.loc[:, var] = (
        data_PD_sin_inconsistencias
        .groupby('fox_insight_id')[var]
        .transform('first')
    )
    
data_PD_2=data_PD_sin_inconsistencias.replace({None: np.nan})

#Para homogeneizar la muestra, se filtra por los pacientes residentes en Estados Unidos:
data_PD_usa=data_PD_2.loc[(data_PD_2['LocCountry']=='United States')].copy()
del data_PD_usa['LocCountry']   




#Creación de TiempoEnf y CoincLat.
    #Tiempo con la enfermedad en el momento de la toma de datos:
data_PD_usa["TiempoEnf"]=data_PD_usa["Edad"]-data_PD_usa["InitPDDiagAge"]


    #Coincidencia lateralidad:
data_PD_usa["CoincLat"] = data_PD_usa.apply(
    lambda row: (
        "Sí" if (
            (row["Lateralidad"] == "Diestro" and row["LateralidadPD"] == "Lado derecho") or 
            (row["Lateralidad"] == "Zurdo" and row["LateralidadPD"] == "Lado izquierdo")
        ) else "No"
    ) if pd.notna(row["Lateralidad"]) and pd.notna(row["LateralidadPD"]) else np.nan,
    axis=1
)


#Eliminamos la variable intermedia LateralidadPD que no entrará en los modelos:
del data_PD_usa["LateralidadPD"]



#----------------------------------------------------------------------------------------------------------------------------#

#TOMAR REGISTRO ÚNICO POR PACIENTE:
#Criterio: registro con mayor grado de información, y en caso de empate el más reciente.

info_cols = [c for c in data_PD_usa.columns if c not in ['fox_insight_id', 'Edad']]


data_PD_usa['info_count'] = data_PD_usa[info_cols].notna().sum(axis=1)

data_PD_ord = data_PD_usa.sort_values(
    by=['fox_insight_id', 'info_count', 'Edad'],
    ascending=[True, False, False]
)


data_PD_unicos = (
    data_PD_ord
    .drop_duplicates(subset='fox_insight_id', keep='first')
    .drop(columns=['info_count'])  # eliminar columna auxiliar
    .reset_index(drop=True)
)



#----------------------------------------------------------------------------------------------------------------------------#

#INCOSISTENCIAS OBVIAS:
data_PD_unicos=data_PD_unicos[~(data_PD_unicos['InitPDDiagAge']<5)]
data_PD_unicos=data_PD_unicos[~(data_PD_unicos['Altura(in)']<27)]

#Paciente con proporción irreal de altura y peso:
data_PD_unicos=data_PD_unicos[~(data_PD_unicos['fox_insight_id']=='FOX_749101')]


#Eliminamos la variable intermedia EdadDiag que no entrará en los modelos:
del data_PD_unicos["InitPDDiagAge"]


#----------------------------------------------------------------------------------------------------------------------------#

#TRATAMIENTO DE MISSINGS

predictores = [col for col in data_PD_unicos.columns if col not in ['fox_insight_id', 'CalidadVida']]

missing_percent = data_PD_unicos[predictores].isnull().mean() * 100
print("Porcentajes de missings:\n",missing_percent.sort_values(ascending=False))

cols_pocos_missing = missing_percent[missing_percent <= 10].index.tolist()
data_PD_limpio = data_PD_unicos.dropna(subset=cols_pocos_missing)


print("\nDistribución de clases:\n",data_PD_limpio['CalidadVida'].value_counts())


missing_percent_2 = data_PD_limpio[predictores].isnull().mean() * 100


var_missings_altos=missing_percent[missing_percent > 10].index.tolist()

#Vemos su distribucón por clases
for var in var_missings_altos:
    print(f"\nDistribución de {var} por clase en 'CalidadVida':")

    # Tabla cruzada absoluta
    tabla_abs = pd.crosstab(data_PD_limpio[var], data_PD_limpio['CalidadVida'])
    tabla_pct = pd.crosstab(data_PD_limpio[var], data_PD_limpio['CalidadVida'], normalize='columns') * 100
    
    print(tabla_abs)
    print(tabla_pct.round(2))
    

#Realizamos la imputación para estas variables

# Copia del DataFrame original para trabajar con él
data_imp = data_PD_limpio.copy()

# Codificar variables binarias manualmente:
bin_map1 = {'No': 0, 'Sí': 1}
bin_map2 = {'Diestro': 0, 'Zurdo': 1}
bin_map3 = {'No': 0, 'Sí': 1}
bin_map4 = {'No/Desconocido': 0, 'Sí': 1}

data_imp['CoincLat'] = data_imp['CoincLat'].map(bin_map1)
data_imp['Lateralidad'] = data_imp['Lateralidad'].map(bin_map2)
data_imp['Pobreza'] = data_imp['Pobreza'].map(bin_map3)
data_imp['HistFamPD'] = data_imp['HistFamPD'].map(bin_map4)



# Verifica que no haya strings restantes
assert all(data_imp[var].dropna().apply(lambda x: isinstance(x, (int, float))).all() for var in var_missings_altos), \
    "Error: aún quedan strings en las variables binarias."

# Columnas que no deben usarse como predictores
excluir = ['fox_insight_id', 'CalidadVida'] + var_missings_altos
predictores = [col for col in data_imp.columns if col not in excluir]

# --- Paso 1: Preparar predictores ---
cat_cols = data_imp[predictores].select_dtypes(include=['object','category']).columns.tolist()
num_cols = [col for col in predictores if col not in cat_cols]

# Codificar categóricas
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
data_imp[cat_cols] = encoder.fit_transform(data_imp[cat_cols].astype(str))

# Escalar numéricos
scaler = MinMaxScaler()
data_imp[num_cols] = scaler.fit_transform(data_imp[num_cols])

# Preparar datos para imputación
columnas_modelo = predictores + var_missings_altos
df_modelo = data_imp[columnas_modelo].copy()

# --- Paso 2: Imputación ---
imputer = KNNImputer(n_neighbors=3)
df_array_imputada = imputer.fit_transform(df_modelo)

# Reconstrucción del DataFrame
df_imputado = pd.DataFrame(df_array_imputada, columns=columnas_modelo, index=data_imp.index)

# --- Paso 3: Invertir transformaciones ---
# Inversión del escalado
df_imputado[num_cols] = scaler.inverse_transform(df_imputado[num_cols])

# Inversión de categóricas
df_imputado[cat_cols] = np.round(df_imputado[cat_cols]).astype(int)
df_imputado[cat_cols] = encoder.inverse_transform(df_imputado[cat_cols])

# Restaurar variables binarias a su forma original ('Sí' / 'No')
reverse_bin_map1 = {0: 'No', 1: 'Sí'}
reverse_bin_map2 = {0: 'Diestro', 1: 'Zurdo'}
reverse_bin_map3 ={0: 'No', 1: 'Sí'}
reverse_bin_map4 = {0: 'No/Desconocido', 1: 'Sí'}

df_imputado['CoincLat'] = np.round(df_imputado['CoincLat']).astype(int).map(reverse_bin_map1).astype('category')
df_imputado['Lateralidad'] = np.round(df_imputado['Lateralidad']).astype(int).map(reverse_bin_map2).astype('category')
df_imputado['Pobreza'] = np.round(df_imputado['Pobreza']).astype(int).map(reverse_bin_map3).astype('category')
df_imputado['HistFamPD'] = np.round(df_imputado['HistFamPD']).astype(int).map(reverse_bin_map4).astype('category')


# --- Paso 4: DataFrame final ---
otras_col = data_PD_limpio[['fox_insight_id', 'CalidadVida']]
data_PD_imputado = pd.concat([otras_col, df_imputado], axis=1)


for var in var_missings_altos:
    print(f"\nDistribución de {var} por clase en 'CalidadVida' tras imputación:")

    # Tabla cruzada absoluta
    tabla_abs = pd.crosstab(data_PD_imputado[var], data_PD_imputado['CalidadVida'])
    tabla_pct = pd.crosstab(data_PD_imputado[var], data_PD_imputado['CalidadVida'], normalize='columns') * 100
    
    print(tabla_abs)
    print(tabla_pct.round(2))
    



#----------------------------------------------------------------------------------------------------------------------------#

def hb_outliers_with_zeros(y, C, mark_zero_as_outlier=False):
    """
    Método Hidiroglou–Berthelot para detección de outliers, adaptado a ceros clínicamente válidos.

    Parámetros:
    ----------
    y : array-like
        Variable cuantitativa positiva, con posibles ceros significativos.
    C : float
        Coeficiente multiplicador para definir los límites (default=4).
    mark_zero_as_outlier : bool
        Si True, los ceros (con score infinito) se marcan como outliers.

    Devuelve:
    --------
    outliers : np.ndarray (bool)
        True si el valor es atípico según el método.
    scores : np.ndarray (float)
        Scores HB de cada observación.
    """
    y = np.asarray(y)
    is_zero = y == 0
    is_pos = y > 0
    y_pos = y[is_pos]

    if len(y_pos) == 0:
        raise ValueError("Todos los valores son cero; el método requiere al menos un valor positivo.")

    # Mediana de los valores positivos
    med = np.median(y_pos)

    # Scores HB
    scores = np.empty_like(y, dtype=float)
    scores[is_pos] = np.where(
        y_pos < med,
        1 - med / y_pos,
        y_pos / med - 1
    )
    scores[is_zero] = np.inf

    # Cálculo de límites (fences)
    valid_scores = scores[np.isfinite(scores)]
    q1, q2, q3 = np.percentile(valid_scores, [25, 50, 75])
    d_inf = max(q2 - q1, 0.05 * q2)
    d_sup = max(q3 - q2, 0.05 * q2)
    L = q2 - C * d_inf
    U = q2 + C * d_sup

    # Outliers
    outliers = np.zeros_like(y, dtype=bool)
    outliers[np.isfinite(scores)] = (scores[np.isfinite(scores)] < L) | (scores[np.isfinite(scores)] > U)

    if mark_zero_as_outlier:
        outliers[is_zero] = True

    return outliers, scores



def hb_outliers_upper_only(y, C):
    """
    Método HB modificado: ignora ceros y detecta solo outliers por exceso.

    Parámetros:
    ----------
    y : array-like
        Variable cuantitativa (ceros permitidos pero no analizados).
    C : float
        Coeficiente de ajuste para los límites.

    Devuelve:
    --------
    outliers : np.ndarray (bool)
        True si el valor se considera outlier superior.
    scores : np.ndarray
        Score HB de cada valor (np.nan si el valor es cero).
    """
    y = np.asarray(y)
    mask_pos = y > 0
    y_pos = y[mask_pos]

    if len(y_pos) == 0:
        raise ValueError("No hay valores positivos para analizar.")

    med = np.median(y_pos)

    # Score HB solo para valores positivos
    scores_pos = np.where(y_pos < med, 1 - med / y_pos, y_pos / med - 1)

    # Limitar el cálculo a score superior (outliers por exceso)
    q2 = np.median(scores_pos)
    q3 = np.percentile(scores_pos, 75)
    d_sup = max(q3 - q2, 0.05 * q2)
    upper_limit = q2 + C * d_sup

    # Preparar el array de scores completo
    scores = np.full_like(y, fill_value=np.nan, dtype=float)
    scores[mask_pos] = scores_pos

    # Outliers: solo superior
    outliers = np.zeros_like(y, dtype=bool)
    outliers[mask_pos] = scores_pos > upper_limit

    return outliers, scores


#----------------------------------------------------------------------------------------------------------------------------#




#SIMETRÍA Y OUTLIERS POR CLASES PARA VARIABLES CUANTITATIVAS
variables_cuant=['Edad','Altura(in)','Peso(lb)','TiempoEnf']


data_pos=data_PD_imputado[data_PD_imputado['CalidadVida']=='No aceptable']
    #Clase negativa
data_neg=data_PD_imputado[data_PD_imputado['CalidadVida']=='Aceptable']


#Simetría
g1_neg=data_neg[variables_cuant].skew()
g1_pos=data_pos[variables_cuant].skew()  

print("Coeficiente de asimetría (clase Aceptable):\n", g1_neg)
print("\nCoeficiente de asimetría (clase No aceptable):\n", g1_pos)



skew_threshold = 1

# Diccionario para guardar outliers
outliers_dict = {}
tabla_outliers = []

for var in variables_cuant:
    outliers_dict[var] = {}
    
    for clase, datos, g1 in [('Aceptable', data_neg, g1_neg), ('No aceptable', data_pos, g1_pos)]:
        serie = datos[var].dropna()
        g1_val = g1[var]

        if abs(g1_val) <= skew_threshold:
            # IQR clásico sobre variable original
            Q1 = serie.quantile(0.25)
            Q3 = serie.quantile(0.75)
            IQR = Q3 - Q1

            lower1 = Q1 - 1.5 * IQR
            upper1 = Q3 + 1.5 * IQR
            mask1 = (serie < lower1) | (serie > upper1)

            lower2 = Q1 - 3 * IQR
            upper2 = Q3 + 3 * IQR
            mask2 = (serie < lower2) | (serie > upper2)

            # Índices
            influyentes = serie[mask2].index
            no_influyentes = serie[mask1].index.difference(influyentes)
         
            
        else:
            # --- Método HB adaptado para variables asimétricas ---
            outliers_mask, _ = hb_outliers_upper_only(serie.values, C=16)
            influyentes = serie.index[outliers_mask]
            no_influyentes = []  # HB no considera no influyentes

        # Guardar outliers influyentes para esta variable y clase
        outliers_dict[var][clase] = influyentes
        
        # Guardar en la tabla
        tabla_outliers.append({
            'Variable': var,
            'Clase': clase,
            'Outliers influyentes': len(influyentes),
            'Outliers no influyentes': len(no_influyentes)
        })
        

outlier_indices_total = set()

for variable in outliers_dict:
    for clase in outliers_dict[variable]:
        outlier_indices_total.update(outliers_dict[variable][clase])

outliers_df = data_PD_imputado.loc[list(outlier_indices_total)]
tabla_outliers_df = pd.DataFrame(tabla_outliers)
print(tabla_outliers_df)

# Eliminar outliers del DataFrame original
data_PD_def = data_PD_imputado.drop(index=outlier_indices_total).reset_index(drop=True)

#----------------------------------------------------------------------------------------------------------------------------#




#NORMALIDAD Y MATRIZ DE CORRELACIÓN DE VARIABLES CUANTITATIVAS
normalidad={}
for var in variables_cuant:
    x=data_PD_def[var]
    # Test de normalidad de Kolmogorov-Smirnov
    ks_stat, p_ks = stats.kstest(x, 'norm', args=(x.mean(), x.std()))
    if p_ks>0.05:
        normalidad[var]={'Sí'}
    else:
        normalidad[var]={'No'}


#No hay normalidad para las variables cuantitativas, por lo que aplicamos el método de Spearman para la correlación:
matriz_correlacion=data_PD_def[variables_cuant].corr(method="spearman")
print("\nMatriz de correlación:\n", matriz_correlacion)


#frec_respuesta=data_unicos["Calidad_Vida"].value_counts()



#NORMALIDAD POR GRUPOS Y TEST DE HIPÓTESIS PARA VARIABLES CUANTITATIVAS:

normalidad_grupo={}
p_valores_cuant={}
for var in variables_cuant:
    grupo_pos = data_PD_def[data_PD_def['CalidadVida'] == 'No aceptable'][var]
    grupo_neg = data_PD_def[data_PD_def['CalidadVida'] == 'Aceptable'][var]
    
    # Test de normalidad de Kolmogorov-Smirnov
    ks_stat_pos, p_ks_pos = stats.kstest(grupo_pos, 'norm', args=(grupo_pos.mean(), grupo_pos.std()))
    ks_stat_neg, p_ks_neg = stats.kstest(grupo_neg, 'norm', args=(grupo_neg.mean(), grupo_neg.std()))
    
    if p_ks_pos>0.05 and p_ks_neg>0.05:
        normalidad_grupo[var]={'Sí'}
    
    else:
        normalidad_grupo[var]={'No'}
        test_stat, p_value = stats.mannwhitneyu(grupo_pos, grupo_neg, alternative='two-sided')
        test_name = "Test de Mann-Whitney"    
    
    p_valores_cuant[var] = {'P-valor': p_value}
 
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(var)
    ax[0].hist(grupo_pos, bins=50, color='blue')
    ax[1].hist(grupo_neg, bins=50, color='red')
    
del fig, ax
del ks_stat_pos, p_ks_pos, ks_stat_neg, p_ks_neg, p_value


print("\Test Mann-Whitney:\n", p_valores_cuant)



#TEST CHI-CUADRADO PARA VARIABLES CUALITATIVAS
variables_cualitativas = ['Sexo', 'Raza', 'Veterano', 'Lateralidad', 'CoincLat',
                          'MedicamentosPD', 'OtrosTratPD', 'HistFamPD',
                          'EstudiosUni', 'Pobreza', 'Empleo']

p_valores_cual = pd.DataFrame(index=variables_cualitativas)
for var in variables_cualitativas:
    tabla_contingencia = pd.crosstab(data_PD_def[var], data_PD_def['CalidadVida'])
    #Test chi-chuadrado
    _, p_valor, _, _ = stats.chi2_contingency(tabla_contingencia)
    # Almacenar p-valor en la tabla
    p_valores_cual.loc[var, 'CalidadVida'] = p_valor

p_valores_cual = p_valores_cual.astype(float)
p_valores_cual =  p_valores_cual.sort_values(by='CalidadVida', ascending=True)
#tabla_contingencia = pd.crosstab(data_limpio['LateralidadPD'], data_limpio['CoincLat'])

print("\Test chi-cuadrado:\n", p_valores_cual)


print("\Balanceo de clases:\n", data_PD_def["CalidadVida"].value_counts())



#data_PD_def.to_csv('C:/Users/david/OneDrive/Documentos/TFG MATEMÁTICAS/data_definitivo_tfg_vf.csv', index=False, encoding="utf-8")




