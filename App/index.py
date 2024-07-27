import sys, os
import pickle as pkl
sys.path.insert(0,'/Lib')

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd

from etl import ETL, REQUIRED_FILES
from dataAnalysis import ANALISYS, VARIABLES_SELECCIONADAS_PATH
from variableForecastingForPrediction import trainSarimaxes, ARIMAS_PATH
from randomForestClassifiers import trainRandomForestRegressors, FOREST_REGRESSORS_PATH

import sarimax_utils as su


def cleanUp():
    global status
    status = "cleaningUp"
    for root, dirs, files in os.walk("/Data", topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

if os.environ.get("CLEANUP"):
    cleanUp()

def LoadOrTrainModels():
    global status
    status = "loading"
    if any(not os.path.exists(file) for file in REQUIRED_FILES):
        status = "loadingData"
        ETL()
    status = "loading"

    if any(not os.path.exists(file) for file in [VARIABLES_SELECCIONADAS_PATH]):
        status = "loadingData"
        ANALISYS()
    status = "loading"

    if any(not os.path.exists(file) for file in [ARIMAS_PATH]):
        status = "training"
        sarimax_clima = trainSarimaxes()
    else:
        with open(ARIMAS_PATH, "rb") as f:
            sarimax_clima = pkl.loads(f.read())
    status = "loading"

    if (any(not os.path.exists(file) for file in [FOREST_REGRESSORS_PATH])):
        status = "training"
        forestRegressors = trainRandomForestRegressors()
    else:
        with open(FOREST_REGRESSORS_PATH, "rb") as f:
            forestRegressors = pkl.loads(f.read())
    return {"sarimax_clima": sarimax_clima, "forestRegressors": forestRegressors}

RESULTADOS_DIR = "/Data/results"
RESULTADOS_CLIMA_PATH = f"{RESULTADOS_DIR}/clima_predicciones.pkl"
RESULTADOS_COSECHAS_PATH = f"{RESULTADOS_DIR}/cosechas_predicciones.pkl"
NO_AÑOS = 5

os.makedirs(RESULTADOS_DIR, exist_ok=True)

def loadorGeneratePredictions(models):
    global status
    status = "ready"
    variables_seleccionadas = pkl.loads(open(VARIABLES_SELECCIONADAS_PATH, "rb").read())
    try:
        status = "ready"
        with open(RESULTADOS_CLIMA_PATH, "rb") as f:
            clima_predicciones = pkl.loads(f.read())
    except Exception as e:
        status = "predicting"
        sarimax_clima = models["sarimax_clima"]
        clima_predicciones = {variable: su.predict(sarimax_clima[variable], 12*NO_AÑOS) 
            for variable in sarimax_clima
        }
        with open(RESULTADOS_CLIMA_PATH, "wb") as f:
            pkl.dump(clima_predicciones, f)

    try:
        status = "ready"
        with open(RESULTADOS_COSECHAS_PATH, "rb") as f:
            cosechas_predicciones = pkl.loads(f.read())
    except Exception as e:
        status = "predicting"
        predictores_produccion = {}
        for variedad in variables_seleccionadas:
            _variedad = {}
            predictores_produccion[variedad] = _variedad
            for type in ("min value max".split(" ")):
                df = pd.DataFrame()
                for variable in variables_seleccionadas[variedad]:
                    df[variable] = clima_predicciones[variable][type].resample("Y").mean()
                _variedad[type] = df
        
        forestRegressors = models["forestRegressors"]
        cosechas_predicciones = {variedad: 
            {type: forestRegressors[variedad].predict(predictores_produccion[variedad][type])
                for type in ("min", "value", "max")
            }
            for variedad in forestRegressors
        }
        with open(RESULTADOS_COSECHAS_PATH, "wb") as f:
            pkl.dump(cosechas_predicciones, f)
    status = "ready"
    return {"clima_predicciones": clima_predicciones, "cosechas_predicciones": cosechas_predicciones}

models = LoadOrTrainModels()
predictions = loadorGeneratePredictions(models)

app = FastAPI()

status = "ready"

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/clima")
async def resultados_clima():
    response = {
        "status": status,
    }
    if status != "ready":
        return JSONResponse(content=response, status_code=202)
    clima_predicciones = predictions["clima_predicciones"]
    response = {
        **response,
        "results": clima_predicciones,
    }

    return response

@app.get("/cosechas")
async def resultados_cosechas():
    response = {
        "status": status,
    }
    if status != "ready":
        return JSONResponse(content=response, status_code=202)
    cosechas_predicciones = predictions["cosechas_predicciones"]
    response = {
        **response,
        "results": {variedad: 
            {type: list(_variedad[type])
                for type in ("min", "value", "max")
            }
            for variedad,_variedad in cosechas_predicciones.items()
        }
    }
    return response

@app.get("/reentrenar")
async def reentrenar():
    global models, predictions
    cleanUp()
    models = LoadOrTrainModels()
    predictions = loadorGeneratePredictions(models)
    return 
