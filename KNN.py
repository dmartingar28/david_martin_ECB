# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 20:59:14 2025

@author: David
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm

from ISLP.models import (ModelSpec as MS, summarize)
from ISLP import confusion_table
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score, GridSearchCV)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from imblearn.under_sampling import RandomUnderSampler

def specificity_score(y_true, y_pred):
    """Especificidad: TN / (TN + FP)"""
    table = confusion_table(y_pred, y_true)
    tn = table.loc[False, False]
    fp = table.loc[True, False]
    return tn / (tn + fp)


def npv_score(y_true, y_pred):
    """VPN (Valor Predictivo Negativo): TN / (TN + FN)"""
    table = confusion_table(y_pred, y_true)
    tn = table.loc[False, False]
    fn = table.loc[False, True]
    return tn / (tn + fn)

#------------------------------------------------------------------------------------------------------------------#



data = pd.read_csv('C:/Users/david/OneDrive/Documentos/TFG MATEMÁTICAS/data_definitivo_tfg_vf.csv')
del data['fox_insight_id']

for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].astype('category')

allvars = data.columns.drop(['CalidadVida']) #drop(['fox_insight_id', 'Calidad_Vida'])

# Construcción del diseño
design = MS(allvars)

# Transformación del diseño en matriz numérica
X = design.fit_transform(data)

# Variable binaria de salida (booleana)
y = data.CalidadVida == 'No aceptable'  # True/False → 1/0

# Variable binaria de salida
C = data.CalidadVida

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=33)

rus = RandomUnderSampler(sampling_strategy=1.0, random_state=0)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)


#Escalamos las variables cuantitativas a un rango [0,1]:
variables_cuant=['Edad','Altura(in)','Peso(lb)','TiempoEnf']
X_train_cuant=X_train_res[variables_cuant]
scaler = MinMaxScaler()
X_train_cuant_scaled= scaler.fit_transform(X_train_cuant)

X_train_cuant_scaled_array = scaler.fit_transform(X_train_cuant)
X_train_cuant_scaled = pd.DataFrame(
    X_train_cuant_scaled_array,
    columns=X_train_cuant.columns,
    index=X_train_cuant.index
)

X_train_scaled = pd.concat([X_train_res.drop(columns=X_train_cuant.columns), X_train_cuant_scaled], axis=1)



X_test_cuant=X_test[variables_cuant]
scaler = MinMaxScaler()
X_test_cuant_scaled= scaler.fit_transform(X_test_cuant)

X_test_cuant_scaled_array = scaler.fit_transform(X_test_cuant)
X_test_cuant_scaled = pd.DataFrame(
    X_test_cuant_scaled_array,
    columns=X_test_cuant.columns,
    index=X_test_cuant.index
)

X_test_scaled = pd.concat([X_test.drop(columns=X_test_cuant.columns), X_test_cuant_scaled], axis=1)


#Aplicamos una búsqueda por validación cruzada:
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
knn = KNeighborsClassifier()

# Definir el rango de valores de k a probar
param_grid = {
    'n_neighbors': list(range(1, 31, 2))  # Por ejemplo, k = 1 a 30
}

# Configurar la búsqueda con validación cruzada
grid_search = GridSearchCV(
    knn,
    param_grid,
    cv=cv5,               # Número de folds de la validación cruzada
    scoring='recall', # Métrica de evaluación
    n_jobs=-1           # Usa todos los núcleos disponibles
)






grid_search.fit(X_train_scaled, y_train_res)
best_param=grid_search.best_params_['n_neighbors']


print("'n_neighbors':", best_param)



#Mejor parámetro: k=1:

knn = KNeighborsClassifier(n_neighbors=best_param, weights='distance')
knn.fit(X_train_scaled, y_train_res)


recall = cross_val_score(knn, X_train_scaled, y_train_res, cv=cv5, scoring='recall')

print("\n Recall CV:", np.mean(recall))

knn_pred = knn.predict(X_test_scaled)
print(confusion_table(knn_pred, y_test))


print(f"recall:, {recall_score(y_test, knn_pred):.3f}")
print(f"accuracy:, {accuracy_score(y_test, knn_pred):.3f}")
print(f"especificidad:, {specificity_score(y_test, knn_pred):.3f}")
print(f"VPP:, {precision_score(y_test, knn_pred):.3f}")
print(f"VPN:, {npv_score(y_test, knn_pred):.3f}")
print(f"f1-score:, {f1_score(y_test, knn_pred):.3f}")




