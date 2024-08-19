import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
import pickle as pkl
from functools import reduce
from datetime import datetime
from collections import defaultdict
import os

from dataAnalysis import VARIABLES_SELECCIONADAS_PATH
from etl import DATOS_CLIMA_PATH


MODELS_DIR="/Data/models"
ARIMAS_PATH = f"{MODELS_DIR}/arimas.pkl"


def trainSarimaxes():
    datos_clima =  pd.read_pickle(DATOS_CLIMA_PATH)

    with open(VARIABLES_SELECCIONADAS_PATH, "rb+") as f:
        variables_seleccionadas = pkl.loads(f.read())

    variables_seleccionadas = reduce(lambda s1,s2: s1 | s2, 
        list(set(valores) for valores in variables_seleccionadas.values()), 
        set()
    )

    arimas = dict()
    for variable in variables_seleccionadas:
        data = datos_clima[variable].resample("M").mean(numeric_only=True)
        autoARIMA = auto_arima(data, seasonal=True, m=12)
        autoARIMA.fit(data)
        arimas[variable] = autoARIMA
        print(f"Trained ARIMA for {variable}")
        print(f"{autoARIMA.summary()}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(ARIMAS_PATH, "wb") as f:
        pkl.dump(arimas, f)
    return arimas

