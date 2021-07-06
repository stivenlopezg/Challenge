import os
import json
import pandas as pd
import config.config as cfg
from utilities.data_wrangling import load_data
from alibi_detect.cd import KSDrift, ChiSquareDrift

logger = cfg.logger


def main():
    logger.info("El proceso ha comenzado ... \n")
    files = os.listdir("data/train")
    train_data = pd.concat(objs=[load_data(filepath=f"data/train/{file}") for file in files], axis=0)
    inference_data = load_data(filepath="data/predictions.csv")
    is_drift = {}
    logger.info("Ahora vamos a calcular si hay deriva en los datos nuevos.")
    for col in train_data.columns:
        if type(col) != object:
            reference = train_data[col].to_numpy()
            detector = KSDrift(x_ref=reference, p_val=0.05)
            new_data = inference_data[col].to_numpy()
            is_drift[col] = detector.predict(x=new_data, return_p_val=True, return_distance=True)["data"]
            is_drift[col]["distance"] = float(is_drift[col]["distance"])
            is_drift[col]["p_val"] = float(is_drift[col]["p_val"])
        else:
            reference = train_data[col].to_numpy()
            detector = ChiSquareDrift(x_ref=reference, p_val=0.05)
            new_data = inference_data[col].to_numpy()
            is_drift[col] = detector.predict(x=new_data, return_p_val=True, return_distance=True)["data"]
            is_drift[col]["distance"] = float(is_drift[col]["distance"])
            is_drift[col]["p_val"] = float(is_drift[col]["p_val"])
    logger.info("Se ha terminado de hacer el calculo de deriva.")
    with open("drift_metrics/drift.json", "w") as fp:
        json.dump(is_drift, fp)
    logger.info(f"Se ha guardado en un archivo json correctamente en la ruta: {fp}")
    logger.info("El proceso ha finalizado correctamente.")
    return True


if __name__ == '__main__':
    main()
