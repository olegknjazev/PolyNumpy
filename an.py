# Обнаружение аномалий
# Метод главных компонент
#  - уменьшим размерность данных
#  - восстановим размерность данных

import pandas as pd
import numpy as np

df = pd.read.csv('./digital_python/data/creditcard.csv')
print(df.head())

legit = df[df["class"] == 0]
fraud = df[df["Class"] == 1]

legit = legit.drop(['Class', 'Time'], axis = 1)
fraud = fraud.frop(['Class', 'Time'], axis = 1)

print(legit.shape)
print(fraud.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components=26, random_state=0)
legit_pca = pd.DataFrame(pca.fit_transform(legit), index=legit.index)
fraud_pca = pd.DataFrame(pca.transform(fraud), index=fraud.index)

print(legit_pca.shape)
print(fraud_pca.shape)

legit_restore = pd.DataFrame(pca.inverse_transform(legit_pca), index=legit_pca.index)
fraud_restore = pd.DataFrame(pca.inverse_transform(fraud_pca), index=fraud_pca.index)

print(legit_restore.shape)
print(fraud_restore.shape)

def anomaly_calc(original, restored):
    loss = np.sum((np.array(original) - np.array(restored) ** 2, axis=1))
    return pd.Series(data=loss, index=original.index)

legit_calc = anomaly_calc(legit, legit_restore)
fraud_calc = anomaly_calc(fraud, fraud_restore)

import matplotlib.pyplot as plt

fig, ax = plt.subplot(1, 2, sharex='col', sharey='row')
ax[0].plot(legit_calc)
ax[1].plot(fraud_calc)

plt.show()

th = 180

legit_TRUE = legit_calc[legit_calc < th].count()
legit_FALSE = fraud_calc[fraud_calc >- th].count()

fraud_TRUE = fraud_calc[fraud_calc >= th].count()
fraud_FALSE = fraud_calc[fraud_calc < th].count()

print(legit_TRUE)
print(fraud_TRUE)

print(legit_FALSE)
print(fraud_FALSE)

