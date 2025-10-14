# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


# ===============================================
# Proyecto: Clasificación de riesgo crediticio
# Autor: [Tu nombre]
# Descripción:
#   Este script implementa un flujo completo de machine learning
#   para entrenar, evaluar y guardar un modelo de Random Forest
#   con búsqueda de hiperparámetros mediante GridSearchCV.
# ===============================================

# ---- Librerías principales ----
import os
import json
import gzip
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def cargar_datos(ruta: str) -> pd.DataFrame:

    return pd.read_csv(ruta, compression="zip", index_col=False)


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:

    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df[df["MARRIAGE"] != 0]
    df = df[df["EDUCATION"] != 0]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x < 4 else 4)
    return df


def crear_pipeline() -> Pipeline:

    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]

    transformador = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="passthrough"
    )

    modelo = RandomForestClassifier(random_state=42)

    return Pipeline([
        ("preprocesamiento", transformador),
        ("clasificador", modelo)
    ])


def configurar_gridsearch(pipeline: Pipeline) -> GridSearchCV:
    """
    Define el proceso de búsqueda de hiperparámetros usando GridSearchCV.
    """
    parametros = {
        "clasificador__n_estimators": [50, 100, 200],
        "clasificador__max_depth": [None, 5, 10, 20],
        "clasificador__min_samples_split": [2, 5, 10],
        "clasificador__min_samples_leaf": [1, 2, 4]
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        scoring="balanced_accuracy",
        cv=10,
        n_jobs=-1,
        verbose=2,
        refit=True
    )


def guardar_modelo(ruta: str, modelo):

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with gzip.open(ruta, "wb") as archivo:
        pickle.dump(modelo, archivo)


# =========================================================
# FUNCIONES DE EVALUACIÓN
# =========================================================

def calcular_metricas(nombre: str, y_real, y_pred):

    return {
        "type": "metrics",
        "dataset": nombre,
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "recall": recall_score(y_real, y_pred, zero_division=0),
        "f1_score": f1_score(y_real, y_pred, zero_division=0)
    }


def matriz_confusion_json(nombre: str, y_real, y_pred):

    cm = confusion_matrix(y_real, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": nombre,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])}
    }


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================

def main():

    # --- Carga de datos ---
    train_df = cargar_datos("files/input/train_data.csv.zip")
    test_df = cargar_datos("files/input/test_data.csv.zip")

    # --- Limpieza ---
    train_df = limpiar_datos(train_df)
    test_df = limpiar_datos(test_df)

    # --- Separación de variables ---
    X_train, y_train = train_df.drop(columns="default"), train_df["default"]
    X_test, y_test = test_df.drop(columns="default"), test_df["default"]

    # --- Creación del modelo ---
    pipe = crear_pipeline()
    grid = configurar_gridsearch(pipe)
    grid.fit(X_train, y_train)

    # --- Guardado del modelo ---
    guardar_modelo("files/models/model.pkl.gz", grid)

    # --- Evaluación ---
    y_pred_train = grid.predict(X_train)
    y_pred_test = grid.predict(X_test)

    metricas_train = calcular_metricas("train", y_train, y_pred_train)
    metricas_test = calcular_metricas("test", y_test, y_pred_test)

    cm_train = matriz_confusion_json("train", y_train, y_pred_train)
    cm_test = matriz_confusion_json("test", y_test, y_pred_test)

    # --- Exportar resultados ---
    os.makedirs("../files/output/", exist_ok=True)
    with open("../files/output/metrics.json", "w") as f:
        for item in [metricas_train, metricas_test, cm_train, cm_test]:
            f.write(json.dumps(item) + "\n")


# =========================================================
# PUNTO DE ENTRADA
# =========================================================
if __name__ == "__main__":
    main()
