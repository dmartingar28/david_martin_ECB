# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 20:22:39 2025

@author: David
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, recall_score, confusion_matrix
from ISLP import confusion_table

from ISLP.models import ModelSpec as MS


from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from imblearn.under_sampling import RandomUnderSampler

def specificity_score(y_true, y_pred):
    """Especificidad: TN / (TN + FP)"""
    table = confusion_table(y_pred, y_true)
    tn = table.loc[0, 0]
    fp = table.loc[1, 0]
    return tn / (tn + fp)


def npv_score(y_true, y_pred):
    """VPN (Valor Predictivo Negativo): TN / (TN + FN)"""
    table = confusion_table(y_pred, y_true)
    tn = table.loc[0, 0]
    fn = table.loc[0, 1]
    return tn / (tn + fn)



#------------------------------------------------------------------------------------------------------------------#

class StatsModelsGLMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model_ = None

    def fit(self, X, y):
        X_const = sm.add_constant(X, has_constant='add')
        self.model_ = sm.GLM(y, X_const, family=sm.families.Binomial()).fit()
        return self

    def predict(self, X):
        X_const = sm.add_constant(X, has_constant='add')
        prob = self.model_.predict(X_const)
        return (prob >= 0.5).astype(int)

    def predict_proba(self, X):
        X_const = sm.add_constant(X, has_constant='add')
        prob = self.model_.predict(X_const)
        return np.column_stack([1 - prob, prob])



data = pd.read_csv('C:/Users/david/OneDrive/Documentos/TFG MATEMÁTICAS/data_definitivo_tfg_vf.csv')
del data['fox_insight_id']


# Convertir categóricas a category dtype
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].astype('category')

# Definición de target
response_var = 'CalidadVida'
y = (data[response_var] == 'No aceptable').astype(int)
all_vars = [c for c in data.columns if c != response_var]

# Inicializaciones
remaining_vars = []
excluded_vars = set()
best_score = 0
history = []
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)



specificity_threshold = 0.4  ### Umbral mínimo de especificidad

last_added = None
rus = RandomUnderSampler(sampling_strategy=0.99, random_state=0)

while True:
    improved = False
    best_candidate = None
    best_action = None
    best_vars = list(remaining_vars)
    best_model = None

    print("\n--- Iteración stepwise ---")

    # Intento añadir variables
    for var in [v for v in all_vars if v not in remaining_vars and v not in excluded_vars]:
        test_vars = remaining_vars + [var]
        design = MS(test_vars, intercept=True)
        X = design.fit_transform(data)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                                  stratify=y, random_state=33)
        X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
        model_check = sm.GLM(y_train_res, X_train_res, family=sm.families.Binomial()).fit()
        pvals = model_check.pvalues.drop('Intercept', errors='ignore')
        if (pvals > 0.05).any():
            print(f"Descartada en esta iteración '{var}' por p-valor > 0.05.")
            continue
        wrapper = StatsModelsGLMWrapper()
        recall = cross_val_score(wrapper, X_train_res, y_train_res, cv=cv5,
                                 scoring=make_scorer(recall_score)).mean()
        specificity = cross_val_score(wrapper, X_train_res, y_train_res, cv=cv5,
                                      scoring=make_scorer(specificity_score)).mean()
        print(f"Añadir '{var}': Recall CV = {recall:.4f}, Specificity CV = {specificity:.4f}")
        if recall > best_score and specificity >= specificity_threshold:
            improved = True
            best_score = recall
            best_candidate = var
            best_action = 'add'
            best_vars = test_vars
            best_model = model_check

    # Intento eliminar variables
    for var in list(remaining_vars):
        test_vars = [v for v in remaining_vars if v != var]
        if not test_vars:
            continue
        design = MS(test_vars, intercept=True)
        X = design.fit_transform(data)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                                  stratify=y, random_state=33)
        X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
        model_check = sm.GLM(y_train_res, X_train_res, family=sm.families.Binomial()).fit()
        pvals = model_check.pvalues.drop('Intercept', errors='ignore')
        if (pvals <= 0.05).all():
            wrapper = StatsModelsGLMWrapper()
            recall = cross_val_score(wrapper, X_train_res, y_train_res, cv=cv5,
                                     scoring=make_scorer(recall_score)).mean()
            specificity = cross_val_score(wrapper, X_train_res, y_train_res, cv=cv5,
                                          scoring=make_scorer(specificity_score)).mean()
            print(f"Eliminar '{var}': Recall CV = {recall:.4f}, Specificity CV = {specificity:.4f}")
            if recall > best_score and specificity >= specificity_threshold:
                improved = True
                best_score = recall
                best_candidate = var
                best_action = 'remove'
                best_vars = test_vars
                best_model = model_check

    if improved:
        remaining_vars = best_vars
        last_added = best_candidate if best_action == 'add' else None
        history.append((remaining_vars.copy(), best_score))
        print(f"\n✅ {best_action} '{best_candidate}' -> Recall {best_score:.4f}")
        print(best_model.summary2())  # <-- Aquí sí imprimimos el resumen
        continue

    # Eliminación por p-valores altos
    if remaining_vars:
        design = MS(remaining_vars, intercept=True)
        X = design.fit_transform(data)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                                  stratify=y, random_state=33)
        X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
        model_check = sm.GLM(y_train_res, X_train_res, family=sm.families.Binomial()).fit()
        pvals = model_check.pvalues.drop('Intercept', errors='ignore')
        high_p = [v for v in pvals.index if pvals[v] > 0.05]
        if high_p:
            to_remove = last_added or high_p[0]
            if to_remove in remaining_vars:
                remaining_vars.remove(to_remove)
                excluded_vars.add(to_remove)
                print(f"Eliminada '{to_remove}' por p-valor {pvals[to_remove]:.4f}")
                continue

    print("\nFin stepwise, sin más mejoras.")
    break

# Entrenamiento final y evaluación
print("\n=== Modelo final ===")
print(f"Variables: {remaining_vars}")
print(f"Mejor Recall CV: {best_score:.4f}")

design = MS(remaining_vars, intercept=True)
X_final = design.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, stratify=y, random_state=33)

X_res, y_res = rus.fit_resample(X_train, y_train)
final_model = sm.GLM(y_res, X_res, family=sm.families.Binomial()).fit()

score_final = cross_val_score(StatsModelsGLMWrapper(), X_res, y_res, cv=cv5,
                              scoring=make_scorer(recall_score)).mean()
print(f"Recall CV final: {score_final:.4f}")

probs = final_model.predict(X_test)
preds = (probs >= 0.5).astype(int)

# Matriz de confusión
print(confusion_table(preds, y_test))



print(f"recall:, {recall_score(y_test, preds):.3f}")
print(f"accuracy:, {accuracy_score(y_test, preds):.3f}")
print(f"especificidad:, {specificity_score(y_test, preds):.3f}")
print(f"VPP:, {precision_score(y_test, preds):.3f}")
print(f"VPN:, {npv_score(y_test, preds):.3f}")
print(f"f1-score:, {f1_score(y_test, preds):.3f}")
