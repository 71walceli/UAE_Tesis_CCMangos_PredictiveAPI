import os
import requests
import pandas as pd


API_BASE_URL = "http://django:8080/api"
ETL_DIR = "/Data/etlData"
PRODUCCIONES_PATH = f"{ETL_DIR}/producciones.pickle"
DATOS_CLIMA_PATH = f"{ETL_DIR}/datos_clima.pickle"
REQUIRED_FILES = [PRODUCCIONES_PATH, DATOS_CLIMA_PATH]


def df_change_types(dataframe, column_types: dict):
    for column, type in column_types.items():
        dataframe[column] = dataframe[column].astype(type)
    return dataframe

def dataframe_from_api(session,path):
    return pd.DataFrame(session.get(f"{API_BASE_URL}{path}").json())


def ETL():
    # Extracción de datos
    auth = requests.post(f"{API_BASE_URL}/auth/login/", {
        "username": os.environ["USER"],
        "password": os.environ["PASSWORD"],
    })
    response = auth.json()
    if "access_token" not in response:
        print("status_code:",auth.status_code)
        print("ERROR:",response)
        raise Exception("Error de autenticación")
    session = requests.Session()
    session.headers = {
        "Authorization": f"Bearer {response['access_token']}",
    }

    areas = dataframe_from_api(session,"/lotes")
    producciones = dataframe_from_api(session,"/produccion")
    datos_clima = dataframe_from_api(session,"/weather/data")


    # # Transformación de datos
    # ## Identificación de datos inválidos
    variables_ambientales = [
        "Precipitation",
        "Temp_Air_Mean",
        "Temp_Air_Min",
        "Temp_Air_Max",
        "Dew_Temp_Mean",
        "Dew_Temp_Max",
        "Dew_Temp_Min",
        "Relat_Hum_Mean",
        "Relat_Hum_Min",
        "Relat_Hum_Max",
        "Wind_Speed_Mean",
        "Wind_Speed_Min",
        "Wind_Speed_Max",
        "Atmospheric_Pressure_Max",
        "Atmospheric_Pressure_Min",
    ]
    datos_clima = datos_clima[["Date", *variables_ambientales]]
    años = datos_clima["Date"].map(lambda v: v[:4])

    porcentaje_nulos_anuales = datos_clima.isnull().groupby(años)\
        .aggregate(lambda series: sum(series)/len(series))
    porcentaje_nulos_anuales.loc["Total"] = porcentaje_nulos_anuales\
        .sum(axis=0, numeric_only=True)/porcentaje_nulos_anuales\
        .count(axis=0, numeric_only=True)


    # ## Eliminación de variables
    for columna in porcentaje_nulos_anuales:
        porcentaje = porcentaje_nulos_anuales[columna]["Total"]
        if porcentaje > .02:
            datos_clima.drop(columna, axis=1)
            porcentaje_nulos_anuales.drop(columna, axis=1)
            variables_ambientales.remove(columna)

    producciones = producciones.drop(["FechaRegistro", "Activo", "Id_Usuario", "id"], axis=1)

    # ## Eliminación de registros con datos faltantes

    # ## Conversión de tipos de datos
    datos_clima.set_index(pd.DatetimeIndex(datos_clima["Date"]), inplace=True)
    datos_clima.drop("Date",inplace=True,axis=1)

    producciones = df_change_types(producciones, {
        "Cantidad": float,
        #"Id_Area": int,
        "Id_Lote": int,
    })


    datos_clima = df_change_types(datos_clima, {column: float 
        for column in variables_ambientales
    })
    datos_clima.dtypes


    # ## Integración y unión
    variedades = dataframe_from_api(session,"/variedades")
    lotes = dataframe_from_api(session,"/areas")
    producciones_variedad_id = pd.merge(producciones,lotes[["id","Variedad"]], how='left',left_on="Id_Area",right_on="id")
    producciones_variedad = pd.merge(producciones_variedad_id,variedades[["id","Nombre"]], how='left',left_on="Variedad",right_on="id")
    producciones_variedad["Variedad"] = producciones_variedad["Nombre"]
    producciones_variedad.drop(["id_x","id_y","Nombre"],axis=1,inplace=True)
    producciones = producciones_variedad

    # ## Agrupamiento
    producciones["años"] = producciones["Fecha"].map(lambda fecha: fecha[:4]).astype(int)
    producciones.index = pd.MultiIndex.from_arrays(producciones[["Id_Lote", "años"]].values.T)
    producciones.drop("años", axis=1, inplace=True)

    os.makedirs(ETL_DIR, exist_ok=True)
    producciones.to_pickle(PRODUCCIONES_PATH)
    datos_clima.to_pickle(DATOS_CLIMA_PATH)

