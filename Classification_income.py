#!/usr/bin/env python
# coding: utf-8

# Done by Klego 9/08/2020
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def cambio_valor(valor):
    if valor == '<=50K':
        return 0
    else:
        return 1

ingresos = pd.read_csv('ingresos.csv')
ingresos['income'].unique()
ingresos['income'] = ingresos['income'].apply(cambio_valor)

datos_x = ingresos.drop('income',axis=1)
datos_y = ingresos['income']
X_train, X_test, Y_train, Y_test = train_test_split(datos_x, datos_y, test_size=0.3)

gender = tf.feature_column.categorical_column_with_vocabulary_list("gender",['Female, Male'])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital-status",hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education",hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship",hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native-country",hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass",hash_bucket_size=1000)
age = tf.feature_column.numeric_column("age")
fnlwgt = tf.feature_column.numeric_column("fnlwgt")
educational_num = tf.feature_column.numeric_column("educational-num")
capital_gain = tf.feature_column.numeric_column("capital-gain")
capital_loss = tf.feature_column.numeric_column("capital-loss")
hours_per_week = tf.feature_column.numeric_column("hours-per-week")

columnas_categorias = [gender,occupation,marital_status,education,relationship,native_country,workclass,age,fnlwgt,educational_num,capital_gain,capital_loss,hours_per_week]

funcion_entrada = tf.estimator.inputs.pandas_input_fn(x=X_train, y=Y_train, batch_size=100, num_epochs=None, shuffle=True)

modelo = tf.estimator.LinearClassifier(feature_columns=columnas_categorias)

modelo.train(input_fn=funcion_entrada, steps=8000)

funcion_prediccion = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)

generador_predicciones = modelo.predict(input_fn=funcion_prediccion)

predicciones = list(generador_predicciones)

predicciones_finales = [prediccion['class_ids'][0] for prediccion in predicciones]

print(classification_report(Y_test, predicciones_finales))
