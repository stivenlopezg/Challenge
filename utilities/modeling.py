import joblib
import numpy as np
import pandas as pd
import config.config as cfg
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utilities.custom_pipeline import ColumnsSelector, GetDataFrame, GetDummies

logger = cfg.logger


def cross_validation(data: pd.DataFrame or np.ndarray, label: pd.Series or np.ndarray,
                     estimators: list, scoring: str, **kwargs):
    """
    Validacion cruzada con diferentes estimadores para estimar una metrica de interes
    :param data: DataFrame con las variables independientes
    :param estimators: lista con los estimadores
    :param label: variable dependiente
    :param scoring: metrica de interes
    :return: result
    """
    result = pd.DataFrame(data={name: cross_val_score(estimator=estimator,
                                                      X=data, y=label,
                                                      scoring=scoring,
                                                      **kwargs) for name, estimator in estimators})
    return result


def create_preprocess_pipeline(numerical_features: list, categorical_features: list,
                               features: str = "both", ohe_features: list = None):
    """
    Instancia un Pipeline de transformadores personalizados para realizar el preprocesamiento de los datos
    :param numerical_features: variables numericas
    :param categorical_features: variables categoricas
    :param features: si se usan ambos tipos de variables o solo numericas
    :param ohe_features: nombre de columnas creadas despues de codificar en dummies
    :return: preprocessing
    """
    if features not in ["numeric", "both"]:
        raise ValueError("Error: features debe ser both o numeric.")
    if features == "both":
        if ohe_features is None:
            raise ValueError("Error: cuando features es both debe pasarse el parametro ohe_features.")
        else:
            numeric_preprocessing = Pipeline(steps=[('numeric_selector', ColumnsSelector(columns=numerical_features)),
                                                    ('knn_imputer', KNNImputer()),
                                                    ('robust_scaler', RobustScaler()),
                                                    ('numeric_df', GetDataFrame(columns=numerical_features))])

            categoric_preprocessing = Pipeline(
                steps=[('categorical_selector', ColumnsSelector(columns=categorical_features)),
                       ('ohe', GetDummies(columns=categorical_features))])

            preprocessing = Pipeline(steps=[('feature_union',
                                             FeatureUnion(transformer_list=[("numeric", numeric_preprocessing),
                                                                            ("categoric", categoric_preprocessing)])),
                                            ('dataframe', GetDataFrame(columns=numerical_features + ohe_features))])
            return preprocessing
    else:
        preprocessing = Pipeline(steps=[('numeric_selector', ColumnsSelector(columns=numerical_features)),
                                        ('knn_imputer', KNNImputer()),
                                        ('robust_scaler', RobustScaler()),
                                        ('numeric_df', GetDataFrame(columns=numerical_features))])
        return preprocessing


def save_model(estimator, filepath: str):
    """
    Guarda un modelo previamente entrenado en la ruta proporcionada
    :param estimator: modelo
    :param filepath: ruta
    :return: NoneType
    """
    return joblib.dump(value=estimator, filename=filepath)


def load_model(filepath):
    """
    Carga un objeto desde una ruta proporcionada
    :param filepath: ruta
    :return: model
    """
    model = joblib.load(filename=filepath)
    logger.info("Se ha cargado el modelo correctamente.")
    return model


def variance_inflation(data_transformed: pd.DataFrame):
    """
    Calcula el VIF para un DataFrame
    :param data_transformed: DataFrame preprocesado
    :return: vif
    """
    vif = pd.DataFrame()
    vif["feature"] = data_transformed.columns
    vif["VIF"] = [variance_inflation_factor(data_transformed.values, i) for i in range(len(data_transformed.columns))]
    return vif


def prediction(estimator, data: pd.DataFrame, probability: bool = True):
    """
    Crea una columna nueva sobre un DataFrame con la prediccion de la probabilidad de que sea una transaccion
    fraudulenta
    :param estimator: modelo
    :param data: DataFrame
    :param probability: si se calcula la probabilidad o la clase
    :return: data
    """
    if probability:
        data["is_fraud"] = np.round(estimator.predict_proba(data)[:, 1], 2)
        logger.info("Se ha hecho la prediccion en datos nuevos.")
        return data
    else:
        data["is_fraud"] = estimator.predict(data)
        logger.info("Se ha hecho la prediccion en datos nuevos.")
        return data
