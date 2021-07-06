import shap
import pandas as pd
import config.config as cfg
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

logger = cfg.logger


def load_data(filepath: str, file_type: str = "csv", **kwargs):
    """
     Carga un archivo csv como un DataFrame
    :param filepath: ruta del archivo
    :param file_type: tipo de archivo, si es csv o excel
    :return: data
    """
    if file_type not in ["csv", "excel"]:
        raise ValueError("Error: el parametro file_type debe ser csv o excel")
    if file_type == "csv":
        data = pd.read_csv(filepath, **kwargs)
        logger.info("Se ha cargado el archivo correctamente.")
        return data
    else:
        data = pd.read_excel(filepath, **kwargs)
        logger.info("Se ha cargado el archivo correctamente.")
        return data


def remove_negative_values(data: pd.DataFrame):
    """
    Elimina observaciones con valores -1 del DataFrame
    :param data: DataFrame
    :return: data
    """
    idx = data[(data["B"] == -1) | (data["S"] == -1)].index.tolist()
    data = data.drop(index=idx, axis=0)
    logger.info("Se han eliminado los valores negativos correctamente.")
    return data


def filter_countries(data: pd.DataFrame):
    """
    Filtra los paises que se usan para la construccion de los modelos
    :param data: DataFrame
    :return: data
    """
    data = data[data["J"].isin(["AR", "BR", "MX", "ES", "US"])]
    logger.info("Se ha seleccionado la informacion de los paises de interes.")
    return data


def drop_col(data: pd.DataFrame, cols_to_drop: list):
    """
    Elimina una o varias columnas de un DataFrame
    :param data: DataFrame
    :param cols_to_drop: columnas a eliminar
    :return: data
    """
    data = data.drop(labels=cols_to_drop, axis=1)
    logger.info(f"Se ha eliminado las variables {', '.join(i for i in cols_to_drop)} correctamente.")
    return data


def convert_to_numeric(data: pd.DataFrame, cols_to_convert: list):
    """
    Reemplaza la coma de miles por vacio y se transforma en un valor numerico para las columnas de interes
    :param data: DataFrame
    :param cols_to_convert: columnas
    :return: data
    """
    for col in cols_to_convert:
        data[col] = data[col].str.replace(",", "").apply(lambda x: float(x))
    logger.info(f"Se ha hecho la correccion en las variables {', '.join(i for i in cols_to_convert)} correctamente.")
    return data


def data_after_imputation(dataframe: pd.DataFrame, column: str,
                          method: str = "knn", estimator_params: dict = None):
    """
    Imputa los datos missing de una columna usando diferentes metodos, por defecto, usando el algoritmo K-NN
    :param dataframe: DataFrame
    :param column: columna a imputar
    :param method: metodo para imputar
    :param estimator_params: parametros de la clase KNNImputer
    :return: dataframe
    """
    if method not in ["median", "mean", "mode", "knn", "iterative"]:
        raise ValueError(f"Error, el metodo {method} no es soportado por la funcion. Los valores posibles son: "
                         f"median, mean, mode y knn.")
    dataframe = dataframe.copy()
    if method == "knn":
        if estimator_params is None:
            knn = KNNImputer()
        else:
            knn = KNNImputer(**estimator_params)
        numerical_features = dataframe.select_dtypes(include="number").columns.tolist()
        dataframe.loc[:, numerical_features] = knn.fit_transform(dataframe[numerical_features])
    elif method == "median":
        dataframe[column] = dataframe[column].fillna(dataframe[column].median())
    elif method == "mean":
        dataframe[column] = dataframe[column].fillna(dataframe[column].mean())
    else:
        dataframe[column] = dataframe[column].fillna(dataframe[column].mode())
    return dataframe


def cross_validation_result(metrics_df: list, names: list):
    """
    Concatena varios DataFrames que tienen los resultados de la validacion cruzada
    :param metrics_df: lista de DataFrames
    :param names: lista de nombres para cada DataFrame
    :return: result
    """
    if len(metrics_df) != len(names):
        raise ValueError("Error: ambos parametros deben tener la misma longitud.")
    result = pd.DataFrame()
    for df, name in zip(metrics_df, names):
        result[f"{name}"] = df.mean()
    return result


def export_data(data: pd.DataFrame, filepath: str):
    """
    Exporta un DataFrame con un archivo csv en la ruta de interes
    :param data: DataFrame
    :param filepath: ruta de interes
    :return: NoneType
    """
    return data.to_csv(filepath, index=False)
