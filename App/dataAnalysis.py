import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import pickle as pkl
import timeseries as ts
from pmdarima.arima import auto_arima
import os

from etl import DATOS_CLIMA_PATH, PRODUCCIONES_PATH


ANALISYS_DIR = "/Data/analysis"
VARIABLES_SELECCIONADAS_PATH = f"{ANALISYS_DIR}/variables_seleccionadas.pkl"


def time_ranges(timeseries):
    return timeseries.index[0], timeseries.index[-1]

def overlapping_range(timeseries1, timeseries2):
    range1 = time_ranges(timeseries1)
    range2 = time_ranges(timeseries2)
    min_range = range1[0] if range1[0] > range2[0] else range2[0]
    max_range = range1[1] if range1[1] < range2[1] else range2[1]
    return min_range, max_range

def sort_columns_by_corr(x,y):
    corr = corr_with_predictor(x,y)
    return pd.DataFrame(corr, x.columns).sort_values(ascending=False,by=0,key=abs)

def argsort_all_columns(data, ascending=True, key=None):
    factor = 0 if not ascending else -len(data)+1
    def _key(value):
        if key is None:
            return value
        return key(value)
    df = pd.DataFrame()
    for i,column in enumerate(data.columns):
        df[column] = (factor + data[column].map(_key).argsort()).abs()
    return df

def corr_with_predictor(x,y):
    size = len(x.columns)
    output = np.zeros((size,))
    for i,column in enumerate(x.columns):
        output[i] = np.corrcoef([x[column], y],)[0,1]
    return output

def corr_with_predictors(x,y_matrix, groupby, y_target):
    y_indices = y_matrix[groupby].unique()
    output = np.zeros((len(x.columns),len(y_indices)))
    for i,y_index in enumerate(y_indices):
        output[:,i] = corr_with_predictor(x,y_matrix[y_matrix[groupby] == y_index][y_target])
    return pd.DataFrame(output, index=x.columns, columns=y_indices)


def sarimax_predict(sarimax, n_periods):
    forecast_auto, conf_int_auto = sarimax.predict(n_periods=12, return_conf_int=True)
    df = pd.DataFrame(
        {
            "min": conf_int_auto.T[0],
            "value": forecast_auto.values,
            "max": conf_int_auto.T[1],
        },
        index=forecast_auto.index
    )
    return df

def show_matrix(data, x_tick_labels=None, y_tick_labels=None, colormap="RdBu"):
    fig, ax = plt.subplots(figsize=(13, 8), ncols=1)
    pos = ax.imshow(data, cmap=colormap, interpolation='none', )
    for (j,i),label in np.ndenumerate(data):
        ax.text(i,j,round(label,3),ha='center',va='center')
    cbar = fig.colorbar(pos, ax=ax, extend='both')
    cbar.minorticks_on()
    if x_tick_labels is not None:
        ax.set_xticks(np.arange(0,len(x_tick_labels)))
        ax.set_xticklabels(x_tick_labels)
    if y_tick_labels is not None:
        ax.set_yticks(np.arange(0,len(y_tick_labels)))
        ax.set_yticklabels(y_tick_labels)
    plt.show()

def sarimax_plot(train, results, test=None):
    # TODO Plot tests. 
    # Get forecast and confidence intervals
    values = results["value"]

    # Plot forecast with training data
    ax = train.plot(figsize=(13,5), color="black")
    values.plot(ax=ax, color="green")
    if test is not None:
        test.plot(ax=ax, color="Red")
    ax.fill_between(results.index, 
        results["min"], results["max"], 
        color='green', alpha=0.5
    )
    plt.legend(
        [
            'Training', 
            'Forecast', 
            "Test", 
            'Confidence'
        ], loc='upper left'
    )
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(alpha=0.5)
    plt.autoscale()
    plt.show()
    return ax


def ANALISYS():
    datos_clima =  pd.read_pickle(DATOS_CLIMA_PATH)
    producciones =  pd.read_pickle(PRODUCCIONES_PATH)

    x = datos_clima.resample("Y").mean()[datetime(2020,1,1):]
    y = producciones["Cantidad"]
    y_por_variedad = producciones.groupby(
        ["Variedad", producciones["Fecha"].apply(lambda f: f[:4])], 
        as_index=False
    ).sum(numeric_only=True)

    groupby = "Variedad"
    y_target = "Cantidad"
    corr_matrix = corr_with_predictors(x, y_por_variedad, groupby, y_target)

    columna = "Tommy Atkins"
    values = corr_matrix[columna].abs().sort_values(ascending=False)[corr_matrix[columna].abs() > 0.66]
    variables_tommy = values.index.values

    columna = "Ataulfo"
    values = corr_matrix[columna].abs().sort_values(ascending=False)[corr_matrix[columna].abs() > 0.66]
    variables_ataulfo = values.index.values

    variables_seleccionadas = {
        "Tommy Atkins": variables_tommy,
        "Ataulfo": variables_ataulfo,
    }
    os.makedirs(ANALISYS_DIR, exist_ok=True)
    with open(VARIABLES_SELECCIONADAS_PATH, "wb") as f:
        f.write(pkl.dumps(variables_seleccionadas))

