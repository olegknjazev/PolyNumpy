# Метод опорных векторов (SCM - support vector machine) - классификация и регрессия
# Разделяющая классификация
# Выбирается линия с максимальным отступом

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC

iris = sns.load_dataset('iris')

print(iris.head())

data = iris[['sepal_lenght', 'petal_lenght', 'species']]
data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

data_df_seposa = data_df[data_df['species'] == 'setosa']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

plt.scatter(data_df_seposa['sepal_lenght'], data_df_seposa['petal_lenght'])
plt.scatter(data_df_versicolor['sepal_lenght'], data_df_versicolor['petal_lenght'])

model = SVC(kernel='linear', C=10000)
model.fit(X, y)

print(model.support_vectors_)

plt.scatter(
    model.support_vectors_[:, 0], 
    model.support_vectors_[:, 1], 
    s=400, facecolor='none', 
    edgecolors='black'
    )

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['spepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length'])

y_p = model.predict(X_p)

X_p['species'] = y_p

X_p_setosa = X_p[X_p['species'] == 'setosa']
X_p_versicolor = X_p[X_p['species'] == 'versicolor']

plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.4)
plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.4)

# ДЗ. Убрать из данных iris часть точек (на которых обучаемся) и убедиться, что на предсказание влияют только опорные вектора

#####

# Загружаем датасет iris
iris = sns.load_dataset('iris')

# Исправляем опечатки в названиях колонок
iris = iris.rename(columns={
    'sepal_length': 'sepal_length',
    'petal_length': 'petal_length'
})

# Выбираем только два класса для бинарной классификации
data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

# Обучаем модель с большим C для жесткой границы
model = SVC(kernel='linear', C=10000)
model.fit(X, y)

# Опорные векторы
support_vectors = model.support_vectors_

# Визуализация исходных данных и опорных векторов
plt.scatter(
    data_df['sepal_length'], 
    data_df['petal_length'], 
    c=y.map({'setosa': 0, 'versicolor': 1}),
    cmap='bwr',
    label='Данные'
)
plt.scatter(
    support_vectors[:, 0], support_vectors[:, 1],
    s=100, facecolors='none', edgecolors='k', label='Опорные вектора'
)

# Удаляем все точки, кроме опорных векторов
mask = np.zeros(len(X), dtype=bool)
for sv in support_vectors:
    # Находим индексы опорных векторов в исходных данных
    idxs = np.where((X.values == sv).all(axis=1))[0]
    mask[idxs] = True

X_reduced = X[mask]
y_reduced = y[mask]

# Обучаем заново на оставшихся точках (только опорных)
model_reduced = SVC(kernel='linear', C=10000)
model_reduced.fit(X_reduced, y_reduced)

# Построение границы принятия решения для новой модели
x_min, x_max = X['sepal_length'].min() - 1, X['sepal_length'].max() + 1
y_min, y_max = X['petal_length'].min() - 1, X['petal_length'].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
grid_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['sepal_length', 'petal_length'])

Z = model_reduced.predict(grid_points)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z == 'setosa', alpha=0.3)

# Визуализация оставшихся точек и границы
plt.scatter(
    X_reduced['sepal_length'], 
    X_reduced['petal_length'], 
    c=y_reduced.map({'setosa': 0, 'versicolor': 1}),
    cmap='bwr',
    edgecolors='k',
    label='Опорные точки'
)

plt.legend()
plt.xlabel('sepal_length')
plt.ylabel('petal_length')
plt.title('Обучение на всех точках vs только опорных')
plt.show()

#####

# В случае, если данные перекрываются, то идеальной границы не существует. У модели существует гиперпараметр, который опрееляет "размытие" отступа

data = iris[['sepal_lenght', 'petal_lenght', 'species']]
data_df = data[(data['species'] == 'virginica') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

data_df_virginica = data_df[data_df['species'] == 'virginica']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

c_value = [[10000, 1000, 100, 10], [1, 0.1, 0.01, 0.001]]

fig, ax = plt.subplots(2, 4, sharex='col', sharey='row')

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length'])

for i in range(2):
    for j in range(4):

        ax[i,j].scatter(data_df_virginica['sepal_lenght'], data_df_virginica['petal_lenght'])
        ax[i,j].scatter(data_df_versicolor['sepal_lenght'], data_df_versicolor['petal_lenght'])

        # Если C больше, то отступ задается "жестко". Чем меньше C, тем отступ становится более "размытым"

        model = SVC(kernel='linear', C=10000)
        model.fit(X, y)

        print(model.support_vectors_)

        ax[i, j].scatter(
            model.support_vectors_[:, 0], 
            model.support_vectors_[:, 1], 
            s=400, facecolor='none', 
            edgecolors='black'
            )

            x1_p = np.linspace(
                min(data_df['sepal_length']), max(data_df['spepal_length']), 100
                )
            x2_p = np.linspace(
                min(data_df['petal_length']), max(data_df['petal_length']), 100
                )

            X_p.drop(columns=['species'])

            y_p = model.predict(X_p)

            X_p['species'] = y_p

            X_p_virginica = X_p[X_p['species'] == 'verginica']
            X_p_versicolor = X_p[X_p['species'] == 'versicolor']

            ax[i, j].scatter(
                X_p_virginica['sepal_length'], X_p_virginica['petal_length'], alpha=0.4
                )
            ax[i, j].scatter(
                X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.4
                )

plt.show(X, y)

# Достоинства
#  - Зависимость от небольшого числа опорных векторов => компактность модели
#  - После обучения предсказания проходят очень быстро
#  - На работу метода влияют ТОЛЬКО точки, находящиеся возле отступов, поэтому методы подходят для многомерных данных

# Недостатки
#  - При большом количестве обучающих образцов могут быть значительные вычислительные затраты
#  - Большая зависимость от размытости C/ Поиск может привести к большим вычислительным затратам
#  - У результатов отсутствует вероятностная интерпритация
