import sys
import logging

logger = logging.getLogger("meli-challenge")
logger.setLevel(logging.INFO)
consoleHandle = logging.StreamHandler(sys.stdout)
consoleHandle.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
consoleHandle.setFormatter(formatter)
logger.addHandler(consoleHandle)

# Project ------------------------------------------------------------------------------------------------------------

new_data_path = "data/new_data.csv"
modelpath_ar = "models/xgboost_ar.joblib"
modelpath_br = "models/logistic_regression_br.joblib"
modelpath_mx = "models/xgboost_mx.joblib"
modelpath_us_es = "models/logistic_regression.joblib"


