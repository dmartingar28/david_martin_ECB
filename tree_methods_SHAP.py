# -*- coding: utf-8 -*-
"""
Created on Wed May  7 23:55:11 2025

@author: David
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ISLP.models import (ModelSpec as MS, summarize)
from ISLP import confusion_table, load_data

import sklearn.model_selection as skm
from sklearn.tree import (DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR, plot_tree, export_text)
from sklearn.metrics import (accuracy_score, log_loss)
from sklearn.ensemble import (RandomForestClassifier as RC, GradientBoostingClassifier as GBC)

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


import shap
#------------------------------------------------------------------------------------------------------------------#





data = pd.read_csv('C:/Users/david/OneDrive/Documentos/TFG MATEMÁTICAS/data_definitivo_tfg_vf.csv')
del data['fox_insight_id']

for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].astype('category')
    

allvars = data.columns.drop(['CalidadVida']) #drop(['fox_insight_id', 'Calidad_Vida'])
# Construcción del diseño
design = MS(allvars, intercept=False)
# Transformación del diseño en matriz numérica
X = design.fit_transform(data)
# Variable binaria de salida (booleana)
y = data.CalidadVida == 'No aceptable'  # True/False → 1/0
# Variable binaria de salida
C = data.CalidadVida

nombres_columnas = list(X.columns)

X_train, X_test, y_train, y_test, L_train, L_test = skm.train_test_split(X, y, C, test_size=0.2, stratify=y, random_state=33)

#Undersampling
rus = RandomUnderSampler(sampling_strategy=0.99, random_state=0)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)    
cv5 = skm.StratifiedKFold(5, random_state=0, shuffle=True)



#BAGGING
bag_best = RC(max_features=None,  n_estimators=200, max_depth=3, random_state=0)

bag_best.fit(X_train_res, y_train_res)
y_hat_bag = bag_best.predict(X_test)

feature_imp_bag = pd.DataFrame({'importance': bag_best.feature_importances_},index=nombres_columnas)
feature_imp_bag = feature_imp_bag.reset_index()
feature_imp_bag.columns = ['Feature', 'Importance']
feature_imp_bag = feature_imp_bag.sort_values(by='Importance', ascending=False)


print(confusion_table(y_hat_bag, y_test))

print(f"recall:, {recall_score(y_test, y_hat_bag):.3f}")
print(f"accuracy:, {accuracy_score(y_test, y_hat_bag):.3f}")
print(f"especificidad:, {specificity_score(y_test, y_hat_bag):.3f}")
print(f"VPP:, {precision_score(y_test, y_hat_bag):.3f}")
print(f"VPN:, {npv_score(y_test, y_hat_bag):.3f}")
print(f"f1-score:, {f1_score(y_test, y_hat_bag):.3f}")

explainer_bag = shap.Explainer(bag_best, X_train_res, model_output='probability')

# 2. Calcular los SHAP values sobre el conjunto de test
shap_values_bag = explainer_bag(X_test)
shap_values_bag_pos = shap_values_bag[:,:,1]

# 3. Crear el gráfico beeswarm (summary plot)
X_italics = X_test.rename(columns=lambda s: fr"$\mathit{{{s}}}$")

shap.summary_plot(
    shap_values_bag_pos,
    X_italics.values,
    feature_names=X_italics.columns,
    plot_type="dot",
    color_bar_label="Valor de la característica",
    show=False)

fig = plt.gcf()
fig.set_size_inches(12, 6)  # ancho x alto en pulgadad

ax = plt.gca()
ax.set_xlabel("Valor de Shapley", fontsize=18)
ax.tick_params(axis='x', labelsize=15)  # ticks eje X más grandes
ax.tick_params(axis='y', labelsize=18)  # ticks eje Y más grandes
cbar_ax = fig.axes[-1]
cbar_ax.set_ylabel("Valor de la característica", rotation=90, labelpad=15, fontsize=18)
cbar_ax.tick_params(labelsize=15)
cbar_ax.set_yticks([0, 1])
cbar_ax.set_yticklabels(['Bajo', 'Alto'], rotation=0, va='center', fontsize=15)

plt.tight_layout()
plt.show()







#RANDOM FOREST
RF_best = RC(max_features='sqrt', n_estimators=400, max_depth=27, random_state=0)

RF_best.fit(X_train_res, y_train_res)
y_hat_RF = RF_best.predict(X_test)


feature_imp_RF = pd.DataFrame({'importance': RF_best.feature_importances_},index=nombres_columnas)
feature_imp_RF = feature_imp_RF.reset_index()
feature_imp_RF.columns = ['Feature', 'Importance']
feature_imp_RF = feature_imp_RF.sort_values(by='Importance', ascending=False)

print(confusion_table(y_hat_RF, y_test))

print(f"recall:, {recall_score(y_test, y_hat_RF):.3f}")
print(f"accuracy:, {accuracy_score(y_test, y_hat_RF):.3f}")
print(f"especificidad:, {specificity_score(y_test, y_hat_RF):.3f}")
print(f"VPP:, {precision_score(y_test, y_hat_RF):.3f}")
print(f"VPN:, {npv_score(y_test, y_hat_RF):.3f}")
print(f"f1-score:, {f1_score(y_test, y_hat_RF):.3f}")


explainer_RF = shap.Explainer(RF_best, X_train_res, model_output='probability')

# 2. Calcular los SHAP values sobre el conjunto de test
shap_values_RF = explainer_RF(X_test)
shap_values_RF_pos = shap_values_RF[:,:,1]

# 3. Crear el gráfico beeswarm (summary plot)
X_italics = X_test.rename(columns=lambda s: fr"$\mathit{{{s}}}$")

shap.summary_plot(
    shap_values_RF_pos,
    X_italics.values,
    feature_names=X_italics.columns,
    plot_type="dot",
    color_bar_label="Valor de la característica",
    show=False)

fig = plt.gcf()
fig.set_size_inches(12, 6)  # ancho x alto en pulgadad

ax = plt.gca()
ax.set_xlabel("Valor de Shapley", fontsize=18)
ax.tick_params(axis='x', labelsize=15)  # ticks eje X más grandes
ax.tick_params(axis='y', labelsize=18)  # ticks eje Y más grandes
cbar_ax = fig.axes[-1]
cbar_ax.set_ylabel("Valor de la característica", rotation=90, labelpad=15, fontsize=18)
cbar_ax.tick_params(labelsize=15)
cbar_ax.set_yticks([0, 1])
cbar_ax.set_yticklabels(['Bajo', 'Alto'], rotation=0, va='center', fontsize=15)

plt.tight_layout()
plt.show()


#BOOSTING
boost_best = GBC(random_state=0, n_estimators=300, max_depth=2, learning_rate=0.001)

boost_best.fit(X_train_res, y_train_res)
y_hat_boost = boost_best.predict(X_test)


feature_imp_boost = pd.DataFrame({'importance': boost_best.feature_importances_},index=nombres_columnas)
feature_imp_boost = feature_imp_boost.reset_index()
feature_imp_boost.columns = ['Feature', 'Importance']
feature_imp_boost = feature_imp_boost.sort_values(by='Importance', ascending=False)


print(confusion_table(y_hat_boost, y_test))

print(f"recall:, {recall_score(y_test, y_hat_boost):.3f}")
print(f"accuracy:, {accuracy_score(y_test, y_hat_boost):.3f}")
print(f"especificidad:, {specificity_score(y_test, y_hat_boost):.3f}")
print(f"VPP:, {precision_score(y_test, y_hat_boost):.3f}")
print(f"VPN:, {npv_score(y_test, y_hat_boost):.3f}")
print(f"f1-score:, {f1_score(y_test, y_hat_boost):.3f}")




explainer = shap.Explainer(boost_best, X_train_res)

# 2. Calcular los SHAP values sobre el conjunto de test
shap_values_boost = explainer(X_test)

# 3. Crear el gráfico beeswarm (summary plot)
X_italics = X_test.rename(columns=lambda s: fr"$\mathit{{{s}}}$")

shap.summary_plot(
    shap_values_boost,
    X_italics.values,
    feature_names=X_italics.columns,
    plot_type="dot",
    color_bar_label="Valor de la característica",
    show=False)

fig = plt.gcf()
fig.set_size_inches(12, 6)  # ancho x alto en pulgadad

ax = plt.gca()
ax.set_xlabel("Valor de Shapley", fontsize=18)
ax.tick_params(axis='x', labelsize=15)  # ticks eje X más grandes
ax.tick_params(axis='y', labelsize=18)  # ticks eje Y más grandes
cbar_ax = fig.axes[-1]
cbar_ax.set_ylabel("Valor de la característica", rotation=90, labelpad=15, fontsize=18)
cbar_ax.tick_params(labelsize=15)
cbar_ax.set_yticks([0, 1])
cbar_ax.set_yticklabels(['Bajo', 'Alto'], rotation=0, va='center', fontsize=15)

plt.tight_layout()
plt.show()
