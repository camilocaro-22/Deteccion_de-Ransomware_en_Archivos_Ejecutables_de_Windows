from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
import tempfile

from metadata_extractor import extract_metadata

# ---------------------------
# Inicializar FastAPI
# ---------------------------

app = FastAPI()

# ---------------------------
# Cargar el modelo
# ---------------------------

model = joblib.load("model.pkl")

# Las columnas en el mismo orden del modelo
FEATURE_COLUMNS = [
    "Machine",
    "DebugSize",
    "DebugRVA",
    "MajorImageVersion",
    "MajorOSVersion",
    "ExportRVA",
    "ExportSize",
    "IatVRA",
    "MajorLinkerVersion",
    "MinorLinkerVersion",
    "NumberOfSections",
    "SizeOfStackReserve",
    "DllCharacteristics",
    "ResourceSize",
    "BitcoinAddresses"
]

# ---------------------------
# Modelo de entrada manual
# ---------------------------


class ManualInput(BaseModel):
    Machine: int
    DebugSize: int
    DebugRVA: int
    MajorImageVersion: int
    MajorOSVersion: int
    ExportRVA: int
    ExportSize: int
    IatVRA: int
    MajorLinkerVersion: int
    MinorLinkerVersion: int
    NumberOfSections: int
    SizeOfStackReserve: int
    DllCharacteristics: int
    ResourceSize: int
    BitcoinAddresses: int


# ---------------------------
# ENDPOINT PRINCIPAL
# ---------------------------

@app.get("/api")
def root():
    return {"message": "API funcionando correctamente"}


# ---------------------------
# PREDICCIÓN PARA ARCHIVOS
# ---------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Extraer metadatos
        data = extract_metadata(tmp_path)

        if data is None:
            return {"error": "No se pudieron extraer metadatos del archivo."}

        # Convertir a DataFrame
        df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

        # Predicción
        pred = model.predict(df)[0]
        label = "benign" if pred == 1 else "ransomware"

        return {
            "prediction": label,
            "features": data
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# PREDICCIÓN MANUAL
# ---------------------------

@app.post("/predict_manual")
def predict_manual(data: ManualInput):

    df = pd.DataFrame([data.dict()], columns=FEATURE_COLUMNS)

    pred = model.predict(df)[0]
    label = "benign" if pred == 1 else "ransomware"

    return {
        "prediction": label,
        "features": data.dict()
    }


# ---------------------------
# SERVIR EL FRONTEND (index.html)
# ---------------------------

# Montar carpeta static
app.mount("/", StaticFiles(directory="static", html=True), name="static")
