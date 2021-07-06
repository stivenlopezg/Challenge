import numpy as np
import shap
import pandas as pd
from sklearn.model_selection import train_test_split


def beta_interpretation(feature, beta):
    """
    Interpretacion de los betas
    :param feature: variable independiente
    :param beta: beta o coeficiente
    :return: string
    """
    return f"Por cada aumento de una unidad en {feature}, la probabilidad de que la transaccion sea fraudulenta es {beta}" \
           f" veces mayor de que la transaccion no sea fraudulenta manteniendo todas las demás variables constantes."


def categorization(observed, predicted):
    """
    Devuelve si la transaccion es un VP, FP, FN, o VN basado en el valor observado y predicho
    :param observed: valor observado
    :param predicted: valor predicho
    :return: string
    """
    if predicted == 1 and observed == 1:
        return "VP"
    elif predicted == 1 and observed == 0:
        return "FP"
    elif predicted == 0 and observed == 0:
        return "VN"
    else:
        return "FN"


def split_train_test_data(data: pd.DataFrame, label: str, stratify: bool = False, **kwargs):
    """
    Particiona los datos en entrenamiento y prueba con o sin muestreo estratificado por las variable objetivo
    :param data: DataFrame con las variables independientes
    :param label: variable dependiente
    :param stratify: si se hace muestreo estratificado
    :return: train_data, test_data, train_label, test_label
    """
    label_data = data.pop(label)
    if stratify:
        train_data, test_data, train_label, test_label = train_test_split(data, label_data,
                                                                          test_size=0.3,
                                                                          stratify=label_data, random_state=42)
    else:
        train_data, test_data, train_label, test_label = train_test_split(data, label_data,
                                                                          test_size=0.3, random_state=42)
    return train_data, test_data, train_label, test_label


def calculate_shap_values(estimator, new_data: pd.DataFrame, data: pd.DataFrame = None, estimator_type: str = "tree"):
    """
    Calcula los valores shap usando un explainer para arboles o modelos lineales
    :param estimator: modelo
    :param data: data de entrenamiento
    :param new_data: data nueva
    :param estimator_type: si es un arbol o modelo lineal
    :return: explainer, shap_values
    """
    if estimator_type not in ["tree", "linear"]:
        raise ValueError("Error: estimator_type debe ser tree o linear.")
    if estimator_type == "tree":
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(new_data)
        return explainer, shap_values
    else:
        if data is None:
            raise ValueError("Error: si estimator_type es linear debe pasar dos DataFrames.")
        else:
            explainer = shap.LinearExplainer(estimator, data)
            shap_values = explainer.shap_values(new_data)
            return explainer, shap_values


def characterization_data(observed: pd.Series, predicted: np.ndarray, data: pd.DataFrame):
    """
    DataFrame con el valor observado, predicho, pais, monto y si es VP, FP, FN, VN
    :param observed: valor observado
    :param predicted: valor predicho
    :param data: DataFrame con las variables independientes
    :return: data
    """
    data = pd.concat(objs=[observed.reset_index(drop=True),
                           pd.Series(predicted, name="prediction"),
                           data.reset_index(drop=True)[["J", "Monto"]]], axis=1)
    data["categorization"] = data[["Fraude", "prediction"]].apply(lambda x: categorization(x.Fraude, x.prediction),
                                                                  axis=1)
    return data
