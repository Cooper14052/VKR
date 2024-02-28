# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:22:44 2024

@author: user
"""

import pandas as pd
import matplotlib as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

path_to_csv_file = '/home/user/Рабочий стол/flow_15sec_all.csv' # Переменная, хранящая путь к файлу csv
loaded_data = pd.read_csv(path_to_csv_file, delimiter=',').round(2) # Загружаем csv в DataFrame и указываем разделитель ; для корректного деления колонок
#print(loaded_data.shape)


X = loaded_data.iloc[:,:-1] # Все значения кроме классов
y = loaded_data.iloc[:,-1] # Классы


# Вывод кол-ва столбцов и строк - проверка
#print(x.shape) 
#print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=33) # Разбиение выборки на части

model = RandomForestClassifier(max_depth=8, min_samples_split=10, random_state=5) # Созаём модель случайного леса

model.fit(X_train, y_train) # Тренировка модели случайный лес


y_pred = model.predict(X_test)
#print(accuracy_score(y_test, y_pred))
#print(cross_val_score(model, X_train, y_train, cv=10))
#y_pred = model.predict(X_test)
#print(classification_report(y_pred, y_test))




#x = loaded_data["time"] # Сохраняем в переменную список со столба из DataFrame
#print(x)
