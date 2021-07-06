import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, \
    precision_score, recall_score, f1_score, brier_score_loss


class ClassificationEvaluator(object):
    """
    Clase para calcular las principales metricas de un problema de clasificacion
    """

    def __init__(self, observed: pd.Series or list, predicted: pd.Series or list):
        self.observed = observed
        self.predicted = predicted
        self.metrics = None

    def generate_report(self):
        """
        Genera un DataFrame con las metricas mas usadas de clasificacion
        :return: pd.DataFrame
        """
        report = np.round(pd.DataFrame(classification_report(y_true=self.observed,
                                                             y_pred=self.predicted, output_dict=True)), 2).T
        return report

    def confusion_matrix(self, class_names: dict = None, **kwargs):
        """
        Devuelve la matriz de confusion como un DataFrame
        :return: pd.DataFrame
        """
        table = np.round(pd.crosstab(index=self.observed, columns=self.predicted,
                                     rownames=['Observed'], colnames=['Predicted'], **kwargs), 2)
        if class_names is None:
            return table
        else:
            mapper = class_names
            table = table.rename(columns=mapper, index=mapper)
            return table

    def calculate_metrics(self):
        """
        Calcula las metricas mas usadas
        :return: dict
        """
        metrics = {
            'roc_auc': np.round(roc_auc_score(y_true=self.observed, y_score=self.predicted), 2),
            'accuracy': np.round(accuracy_score(y_true=self.observed, y_pred=self.predicted), 2),
            'precision': np.round(precision_score(y_true=self.observed, y_pred=self.predicted), 2),
            'recall': np.round(recall_score(y_true=self.observed, y_pred=self.predicted), 2),
            'f1': np.round(f1_score(y_true=self.observed, y_pred=self.predicted), 2)
        }
        self.metrics = metrics
        return self

    def print_metrics(self):
        """
        Imprime un resumen de las metricas calculadas
        :return: string
        """
        if self.metrics is None:
            self.calculate_metrics()
        print(f'El AUC es: {self.metrics["roc_auc"]}')
        print(f'La Exactitud es: {self.metrics["accuracy"]}')
        print(f'La Precision es: {self.metrics["precision"]}')
        print(f'La Exhaustividad es: {self.metrics["recall"]}')
        print(f'El F1 es: {self.metrics["f1"]}')
        return None
