# -*- coding: utf-8 -*-
"""
Created on Sun May 11 13:49:56 2025

@author: David
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots, cm
import sklearn.model_selection as skm
from ISLP import confusion_table

from ISLP.models import (ModelSpec as MS, summarize)
from sklearn.svm import SVC
from ISLP.svm import plot as plot_svm
from sklearn.metrics import RocCurveDisplay

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





data = pd.read_csv('C:/Users/david/OneDrive/Documentos/TFG MATEMÁTICAS/data_definitivo_tfg.csv')
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

X_train, X_test, y_train, y_test, L_train, L_test = skm.train_test_split(X, y, C, test_size=0.2, stratify=y, random_state=33)

rus = RandomUnderSampler(sampling_strategy=0.99, random_state=0)
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








#SUPPORT VECTOR CLASSIFIER
svm_linear = SVC(C=10, kernel='linear', random_state=0)
svm_linear.fit(X_train_scaled, y_train_res)


cv5 = skm.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
grid = skm.GridSearchCV(svm_linear, {'C':[0.01, 0.01, 0.1, 1, 10, 100, 1000]}, 
                        refit=True, cv=cv5, scoring='recall',
                        verbose=1)
grid.fit(X_train_scaled, y_train_res)

results_linear=grid.cv_results_[('mean_test_score')]

best_linear = grid.best_estimator_
y_test_hat_1 = best_linear.predict(X_test_scaled)
print(confusion_table(y_test_hat_1, y_test))

print(f"recall:, {recall_score(y_test, y_test_hat_1):.3f}")
print(f"accuracy:, {accuracy_score(y_test, y_test_hat_1):.3f}")
print(f"especificidad:, {specificity_score(y_test, y_test_hat_1):.3f}")
print(f"VPP:, {precision_score(y_test, y_test_hat_1):.3f}")
print(f"VPN:, {npv_score(y_test, y_test_hat_1):.3f}")
print(f"f1-score:, {f1_score(y_test, y_test_hat_1):.3f}")




#SUPPORT VECTOR MACHINE
    #kernel polinómico:
svm_poly = SVC(kernel="poly", degree=2, C=1, random_state=0)
svm_poly.fit(X_train_scaled, y_train_res)

grid_poly = skm.GridSearchCV(svm_poly, {'C':[0.01, 0.1, 1, 10, 100, 1000],'degree':[2,3,4]},
                             refit=True,cv=cv5,scoring='recall',
                             verbose=1
                             )
grid_poly.fit(X_train_scaled, y_train_res)

results_poly=grid_poly.cv_results_[('mean_test_score')]
best_poly = grid_poly.best_estimator_
params_poly=grid_poly.best_params_

y_test_hat_2 = best_poly.predict(X_test_scaled)
print(confusion_table(y_test_hat_2, y_test))


print(f"recall:, {recall_score(y_test, y_test_hat_2):.3f}")
print(f"accuracy:, {accuracy_score(y_test, y_test_hat_2):.3f}")
print(f"especificidad:, {specificity_score(y_test, y_test_hat_2):.3f}")
print(f"VPP:, {precision_score(y_test, y_test_hat_2):.3f}")
print(f"VPN:, {npv_score(y_test, y_test_hat_2):.3f}")
print(f"f1-score:, {f1_score(y_test, y_test_hat_2):.3f}")




    #kernel radial:
svm_rbf = SVC(kernel="rbf", gamma=1, C=1, random_state=0)
svm_rbf.fit(X_train_scaled, y_train_res)

grid_rbf = skm.GridSearchCV(svm_rbf, {'C':[0.01, 0.1, 1, 10, 100, 1000],'gamma':[0.25, 0.5, 1, 2, 4]},
                            refit=True,cv=cv5,scoring='recall',
                            verbose=1)
grid_rbf.fit(X_train_scaled, y_train_res)

results_rbf=grid_rbf.cv_results_[('mean_test_score')]
best_rbf = grid_rbf.best_estimator_
params_rbf=grid_rbf.best_params_

y_test_hat_3 = best_rbf.predict(X_test_scaled)
print(confusion_table(y_test_hat_3, y_test))

print(f"recall:, {recall_score(y_test, y_test_hat_3):.3f}")
print(f"accuracy:, {accuracy_score(y_test, y_test_hat_3):.3f}")
print(f"especificidad:, {specificity_score(y_test, y_test_hat_3):.3f}")
print(f"VPP:, {precision_score(y_test, y_test_hat_3):.3f}")
print(f"VPN:, {npv_score(y_test, y_test_hat_3):.3f}")
print(f"f1-score:, {f1_score(y_test, y_test_hat_3):.3f}")
