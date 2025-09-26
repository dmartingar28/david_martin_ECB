# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 20:24:04 2025

@author: David
"""

import numpy as np
import pandas as pd


from ISLP.models import ModelSpec as MS
from ISLP import confusion_table
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score)
from sklearn.naive_bayes import GaussianNB, BernoulliNB

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


# 1. Identificar variables cuantitativas y cualitativas
categorical_vars = data.select_dtypes(include='category').columns.drop('CalidadVida').tolist()
quantitative_vars = data.select_dtypes(exclude=['object', 'category']).columns.tolist()

# 2. Definir X y y
allvars = data.columns.drop(['CalidadVida'])

# Diseño
design = MS(allvars, intercept=False)
X = design.fit_transform(data)  # Esto convierte categóricas en binarias (one-hot)
y = data.CalidadVida == 'No aceptable'


# 3. Separar variables codificadas
bin_vars = [col for col in X.columns if any(cat in col for cat in categorical_vars)]
quant_vars = [col for col in X.columns if col not in bin_vars]

X_bin = X[bin_vars]
X_quant = X[quant_vars]

# 4. Separar en train/test
Xq_train, Xq_test, Xb_train, Xb_test, y_train, y_test = train_test_split(
    X_quant, X_bin, y, test_size=0.2, stratify=y, random_state=33
)

# 5. Undersampling
X_train_concat = pd.concat([Xq_train, Xb_train], axis=1)
rus = RandomUnderSampler(sampling_strategy=0.99, random_state=0)
X_train_resampled, y_train_res = rus.fit_resample(X_train_concat, y_train)

# 6. Volver a separar
Xq_train_res = X_train_resampled[quant_vars]
Xb_train_res = X_train_resampled[bin_vars]

# 7. Entrenar modelos
nb_q = GaussianNB()
nb_b = BernoulliNB()

nb_q.fit(Xq_train_res, y_train_res)
nb_b.fit(Xb_train_res, y_train_res)

# 8. Predicción y combinación
log_prob_q = nb_q.predict_log_proba(Xq_test)
log_prob_b = nb_b.predict_log_proba(Xb_test)

log_prob_combined = log_prob_q + log_prob_b
y_pred_test = np.argmax(log_prob_combined, axis=1)



# 9. Evaluación

print(confusion_table(y_pred_test, y_test))

print(f"recall:, {recall_score(y_test, y_pred_test):.3f}")
print(f"accuracy:, {accuracy_score(y_test, y_pred_test):.3f}")
print(f"especificidad:, {specificity_score(y_test, y_pred_test):.3f}")
print(f"VPP:, {precision_score(y_test, y_pred_test):.3f}")
print(f"VPN:, {npv_score(y_test, y_pred_test):.3f}")
print(f"f1-score:, {f1_score(y_test, y_pred_test):.3f}")



# 10. Análisis de medias y probabilidades
df_medias = pd.DataFrame(nb_q.theta_, index=nb_q.classes_, columns=quant_vars)
df_var  = pd.DataFrame(nb_q.var_,   index=nb_q.classes_, columns=quant_vars)
df_std = pd.DataFrame(
    np.sqrt(nb_q.var_),
    index=nb_q.classes_,
    columns=quant_vars
)


df_probs = pd.DataFrame(np.exp(nb_b.feature_log_prob_), index=nb_b.classes_, columns=bin_vars)



