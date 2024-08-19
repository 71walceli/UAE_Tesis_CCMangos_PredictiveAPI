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

    x = {}
    _datos_clima = datos_clima.resample("Y").mean(numeric_only=True)
    for variedad in variables_seleccionadas:
        x[variedad] = _datos_clima[variables_seleccionadas[variedad]].loc["2020-01-01":"2023-12-31"] # TODO Revise hardcode
    

    y = producciones[["Cantidad", "Variedad"]]
    y_variedades = {}
    for variedad in variables_seleccionadas:
        y_variedades[variedad] = y[y["Variedad"] == variedad]["Cantidad"]\
            .groupby(producciones["Fecha"].apply(lambda f: f[:4])).mean(numeric_only=True)
    
    forestRegressors = {
        variedad: RandomForestRegressor().fit(x[variedad], y_variedades[variedad])
        for variedad in variables_seleccionadas
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    pkl.dump(forestRegressors, open(FOREST_REGRESSORS_PATH, "wb"))
    return forestRegressors


