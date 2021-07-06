import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def bar_plot(dataframe: pd.DataFrame, column: str, **kwargs):
    """
    Grafico de barras sobre una columna de interes
    :param dataframe: DataFrame
    :param column: columna de interes
    :return: NoneType
    """
    data = dataframe[column].value_counts().sort_values(ascending=True)
    bars = tuple(data.index.tolist())
    values = data.values.tolist()
    y_pos = np.arange(len(bars))
    colors = ["lightblue"] * len(bars)
    colors[-1] = "blue"
    plt.figure(figsize=(16, 10), **kwargs)
    plt.barh(y_pos, values, color=colors)
    plt.title(f"Distribución de {column}")
    plt.yticks(ticks=y_pos, labels=bars)
    return plt.show()


def box_plot(dataframe: pd.DataFrame, column: str, label: str = None, **kwargs):
    """
    Diagrama de caja y bigotes para ver la distribucion de una variable de interes y de esa variable con respeto
    a una variable categorica
    :param dataframe: DataFrame
    :param column: variable de interes
    :param label: variable categorica (suele ser la variable dependiente si es un problema de clasificacion)
    :return: NoneType
    """
    if label is None:
        return sns.catplot(data=dataframe, x=column, kind="box", **kwargs)
    else:
        return sns.catplot(data=dataframe, x=label, y=column, kind="box", **kwargs)


def distribution_plot(dataframe: pd.DataFrame, column: str, **kwargs):
    """
    Histograma de una columna de interes
    :param dataframe: DataFrame
    :param column: columna de interes
    :return: Nonetype
    """
    return sns.displot(data=dataframe, x=column, **kwargs)


def heat_map(dataframe: pd.DataFrame, **kwargs):
    """
    Matriz de calor sobre un DataFrame
    :param dataframe: DataFrame
    :return:
    """
    return sns.heatmap(dataframe, annot=True, fmt=".1g", linecolor="w", linewidths=3, **kwargs)


def plot_feature_importance(shap_values, data: pd.DataFrame, **kwargs):
    """
    Grafico de barras de la importancia de las variables a partir de los valores shap
    :param shap_values: valores shap
    :param data: DataFrame
    :return:
    """
    return shap.summary_plot(shap_values, data, plot_type="bar", plot_size=(14, 10), **kwargs)
