import numpy as np
import pandas as pd

# Pandas - расширение NumPy (структурированные массивы). Строки и столбцы индексируются метками, а не только числовыми значениями

# Series, DataFrame, Index

## Series

data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data)
print(type(data))

print(data.values)
print(type(data.values))

print(data.index)
print(type(data.index))

data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data[0])
print(data[1:3])

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
print(data)
print(data(['a']))
print(data['b':'d'])

print(type(data.index))

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=[1, 10, 7, 'd'])
print(data)

print(data[1])
print(data[10:'d'])

population_dict = {
    'city_1': 1001,
    'city_2': 1002,
    'city_3': 1003,
    'city_4': 1004,
    'city_5': 1005,
}

population = pd.Series(population_dict)
print(population)

print(population['city_4'])
print(population['city_4':'city_5'])

# Для создания Series можно использовать
# - списки Python или массивы NumPy
# - скалярные значения
# - словари
# 1. Привести различные способы создания объектов типа Series
#####
# Указание индексов вручную
data = [10, 20, 30]
index = ['a', 'b', 'c']
series = pd.Series(data, index=index)
print(series)

# Создание пустого Series
series = pd.Series()
print(series)

# Создание Series из другой структуры данных (например, DataFrame)
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
series = df['A']
print(series)

# Создание Series с использованием функции pd.Series.from_array()
data = [1, 2, 3]
series = pd.Series.from_array(data)
print(series)

# Из объектов типа pd.date_range()
date_range = pd.date_range(start='2023-01-01', periods=5)
series = pd.Series(range(5), index=date_range)
print(series)

#####

## DataFrame - двумерный массив с явно определенными индексами. Последовательность "согласованных объектов Series"

population_dict = {
    'city_1': 1001,
    'city_2': 1002,
    'city_3': 1003,
    'city_4': 1004,
    'city_5': 1005,
}

area_dict = {
    'city_1': 9991,
    'city_2': 9992,
    'city_3': 9993,
    'city_4': 9994,
    'city_5': 9995,
}

population = pd.Series(population_dict)
area = pd.Series(area_dict)

# print(population)
# print(area)

states = pd.DataFrame({
    'population': population,
    'area1': area
})

print(states)

print(type(states.values))
print(type(states.index))
print(type(states.columns))

print(states['area1'])

# DataFrame. Способы создания
# - через объекты Series
# - списки словарей
# - словари объектов Series
# - двумерный массив NumPy
# - структурированный массив NumPy

# 2. Привести различные способы создания объектов типа DataFrame
#####
# Из другого DataFrame
data = {
    'Column1': [1, 2, 3],
    'Column2': [4, 5, 6]
}
df1 = pd.DataFrame(data)
df2 = pd.DataFrame(df1)
print(df2)

# Из скалярного значения
df = pd.DataFrame(5, index=range(3), columns=range(4))
print(df)

# С использованием метода pd.read_csv() (или других функций чтения)
df = pd.read_csv('file.csv')

# Из среза другого DataFrame
data = {
    'Column1': [1, 2, 3, 4],
    'Column2': [5, 6, 7, 8]
}
df1 = pd.DataFrame(data)
df2 = df1[['Column1']]
print(df2)

# Из словаря списков с указанием индексов
data = {
    'Column1': [1, 2, 3],
    'Column2': [4, 5, 6]
}
index = ['A', 'B', 'C']
df = pd.DataFrame(data, index=index)
print(df)

# Из списка списков или кортежей
data = [
    [1, 4],
    [2, 5],
    [3, 6]
]
df = pd.DataFrame(data, columns=['Column1', 'Column2'])
print(df)

#####

# Index - способ организации ссылки на данные объектов Series и DataFrame. Index - неизменяем, упорядочен, является мультимножеством (могут быть повторяющиеся значения)

ind = pd.Index([2,3,5,7,11])
print(ind[1])
print(ind[::2])

# ind[1] = 5

# Index - следует соглашениям объекта set (python)

indA = pd.index([1,2,3,4,5])
indB = pd.index([2,3,4,5,6])

print(indA.intersection(indB))

# Выборка данных из Series

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])

print('a' in data)
print('z' in data)

print(data.keys())

print(list(data.items()))

data['a'] = 100
data['b'] = 1000

print(data)

## как одномерный массив

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])

print(data['a':'c'])
print(data[0:2])
print(data[(data > 0.5) & (data < 1)])
print(data['a', 'd'])

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=[1, 3, 10, 15])

print(data[1])

print(data.loc[1])
print(data.iloc[1])

# Выборка данных из DataFrame
# как словарь

pop = pd.Series({
    'city_1': 1001,
    'city_2': 1002,
    'city_3': 1003,
    'city_4': 1004,
    'city_5': 1005,
})

area = pd.Series({
    'city_1': 9991,
    'city_2': 9992,
    'city_3': 9993,
    'city_4': 9994,
    'city_5': 9995,
})

data = pd.DataFrame({'area1': area, 'pop1': pop})

print(data)

print(data['area1'])
print(data.area1)

print(data.pop1 is data['pop1'])
print(data.pop is data['pop'])

data['new'] = data['area1']
data['new1'] = data['area1'] / data['pop1']

print(data)

# Двумерный NumPy-массив

data = pd.DataFrame({'area1': area, 'pop1': pop, 'pop': pop})

print(data)

print(data.values)

print(data.T)

print(data['area1'])

print(data.values[0:3])

# атрибут-индексаторы


print(data)
print(data.iloc[:3, 1:2])
print(data.loc[:'city_4', 'pop1':'pop'])
print(data.loc[data['pop'] > 1002, ['area1', 'pop']])

data.iloc[0,2] = 999999

print(data)

rng = np.random.default_rng()
s =pd.Series(rng.integers(0,10,4))

print(s)
print(np.exp(s))

pop = pd.Series({
    'city_1': 1001,
    'city_2': 1002,
    'city_3': 1003,
    'city_41': 1004,
    'city_51': 1005,
})

area = pd.Series({
    'city_1': 9991,
    'city_2': 9992,
    'city_3': 9993,
    'city_42': 9994,
    'city_52': 9995,
})

data = pd.DataFrame({'area1': area, 'pop1': pop})

print(data)

# 3. Объедините два объекта Series с неодинаковыми множествами ключей (индексов) так, чтобы вместо NaN было установлено значение 1
#####
combined = pop.add(area, fill_value=1)
print(combined)

#####

dfA = pd.DataFrame(rng.integers(0,10,(2,2)), colums=['a','b'])
dfB = pd.DataFrame(rng.integers(0,10, (3,3)), columns=['a','b','c'])

print(dfA)
print(dfB)

print(dfA + dfB)

rng = np.random.default_rng(1)

A = rng.integers(0,10, (3,4))
print(A)

print(A[0])

print(A - A[0])

df = pd.DataFrame(A, columns=['a','b','c','d'])
print(df)

print(df.iloc[0])

print(df - df.iloc[0])

print(df.iloc[0, ::2])

print(df - df.iloc[0, ::2])

# 4. Переписать пример с транслированием для DataFrame так, чтобы вычитание происходило по СТОЛБЦАМ
#####
A = rng.integers(0, 10, (3, 4))
print(A)
print(A[0])
print(A - A[0])

df = pd.DataFrame(A, columns=['a', 'b', 'c', 'd'])
print(df)

print(df.iloc[0])

print(df.subtract(df.iloc[0], axis=1))

print(df.iloc[0, ::2])

print(df.subtract(df.iloc[0, ::2], axis=1))

#####

# Pandas. Два способа хранения отсутствующих значений
# индикаторы Nan, None
# null

# None - объект (накладные расходы). Не работает с sum, min

val1 = np.array([1,2,3])
print(val1.sum())

val1 = np.array([1,None,2,3])
print(val1.sum())

val1 = np.array([1, np.nan, 2,3])
print(val1)
print(val1.sum())
np.sum(val1)
print(np.nansum(val1))

x =pd.Series(range(10), dtype=int)

x[0] = None
x[1] = np.nan

print(x)

x1 = pd.Series(['a','b','c'])

x1[0] = None
x1[1] = np.nan

print(x1)

x2 = pd.Series([1,2,3,np.nan,None, pd.NA])
print(x2)

x3 = pd.Series([1,2,3,np.nan,None, pd.NA], dtype='int32')
print(x3)

print(x3.isnull())
print(x3[x3.isnull()])
print(x3[x3.notnull()])

print(x3.dropna())

df = pd.DataFrame(
    [
        [1,2,3,np.nan, None, pd.NA],
        [1,2,3,4,5,6],
        [1,np.nan,3,None,np.nan,6]
    ]
)

print(df)
print(df.dropna())
print('dddddd')
print(df.dropna(axis=0))
print(df.dropna(axis=1))

# how
# - all - все значения NA
# - any - хотя бы одно значение
# - thresh = x, остается, если присутствует МИНИМУМ x НЕПУСЫХ значений

print(df.dropna(axis=1, how='all'))
print(df.dropna(axis=1, how='any'))
print(df.dropna(axis=1, thresh=2))

# 5. На примере объектов DataFrame продемонстрировать использование методов ffill() и bfill()
#####
# ffill() (forward fill) заполняет пропуски предыдущими значениями в том же столбце.
data = {
    'A': [1, np.nan, 3, np.nan, 5],
    'B': [np.nan, 2, np.nan, 4, 5],
    'C': [1, 2, 3, 4, 5]
}

df = pd.DataFrame(data)
print(df)

ffilled_df = df.ffill()
print(ffilled_df)

# bfill() (backward fill) заполняет пропуски следующими значениями в том же столбце.
bfilled_df = df.bfill()
print(bfilled_df)

#####