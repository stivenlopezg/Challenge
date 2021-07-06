import warnings
import pandas as pd
import config.config as cfg
from utilities.modeling import load_model, prediction
from utilities.data_wrangling import load_data, remove_negative_values,\
                                     filter_countries, drop_col, convert_to_numeric, export_data


logger = cfg.logger


def main():
    warnings.filterwarnings(action="ignore")
    logger.info("El proceso de prediccion ha comenzado ... \n")
    data = load_data(filepath=cfg.new_data_path, sep=",", decimal=".")
    flag = data[["Q", "R", "Monto"]].dtypes == object
    cols_to_convert = data[["Q", "R", "Monto"]].dtypes[flag].index.tolist()
    if len(cols_to_convert) > 0:
        data = convert_to_numeric(data=data, cols_to_convert=cols_to_convert)
    else:
        pass
    data = remove_negative_values(data=data)
    data = drop_col(data=data, cols_to_drop=["K"])
    data = filter_countries(data=data)
    model_ar = load_model(filepath=cfg.modelpath_ar)
    prediction_ar = prediction(estimator=model_ar, data=data[data["J"].isin(["AR"])])
    model_br = load_model(filepath=cfg.modelpath_br)
    prediction_br = prediction(estimator=model_br, data=data[data["J"].isin(["BR"])])
    model_mx = load_model(filepath=cfg.modelpath_mx)
    prediction_mx = prediction(estimator=model_mx, data=data[data["J"].isin(["MX"])])
    model_us_es = load_model(filepath=cfg.modelpath_us_es)
    prediction_us_es = prediction(estimator=model_us_es, data=data[data["J"].isin(["ES", "US"])])
    prediction_df = pd.concat(objs=[prediction_ar, prediction_br, prediction_mx, prediction_us_es])
    export_data(data=prediction_df, filepath="data/predictions.csv")
    logger.info("Las predicciones se han exportado como un csv correctamente.")
    logger.info("El proceso ha concluido correctamente.")
    return True


if __name__ == '__main__':
    main()
