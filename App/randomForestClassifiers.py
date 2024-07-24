import pandas as pd
import numpy as np
import pickle as pkl
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import os

from etl import PRODUCCIONES_PATH, DATOS_CLIMA_PATH
from dataAnalysis import VARIABLES_SELECCIONADAS_PATH


MODELS_DIR="/Data/models"
FOREST_REGRESSORS_PATH = f"{MODELS_DIR}/forest_regressors.pickle"


def trainRandomForestRegressors():
    datos_clima =  pd.read_pickle(DATOS_CLIMA_PATH)
    producciones =  pd.read_pickle(PRODUCCIONES_PATH)

    # # Prep before training
    with open(VARIABLES_SELECCIONADAS_PATH, "rb+") as f:
        variables_seleccionadas = pkl.loads(f.read())

    #x = datos_clima.resample("Y").mean().drop(["Temp_Air_Mean"], axis=1)
    _datos_clima = datos_clima.resample("Y").mean()
    x_tommy = _datos_clima[variables_seleccionadas["Tommy Atkins"]]
    x_tommy = x_tommy.loc["2020-01-01":"2023-12-31"]

    x_ataulfo = _datos_clima[variables_seleccionadas["Ataulfo"]]
    x_ataulfo = x_ataulfo.loc["2020-01-01":"2023-12-31"]

    y = producciones[["Cantidad", "Variedad"]]
    y_tommy = y[y["Variedad"] == "Tommy Atkins"]["Cantidad"]\
        .groupby(producciones["Fecha"].apply(lambda f: f[:4])).mean()
    y_ataulfo = y[y["Variedad"] == "Ataulfo"]["Cantidad"]#.values

    # # Model Training
    forest_regressor_tommy = RandomForestRegressor().fit(x_tommy, y_tommy)
    forest_regressor_ataulfo = RandomForestRegressor().fit(x_ataulfo, y_ataulfo)
    
    forestRegressors = {
        "Tommy Atkins": forest_regressor_tommy,
        "Ataulfo": forest_regressor_ataulfo
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    pkl.dump(forestRegressors, open(FOREST_REGRESSORS_PATH, "wb"))
    return forestRegressors


