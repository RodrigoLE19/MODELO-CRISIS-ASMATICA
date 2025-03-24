import pickle
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

# Configurar middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Obtén la ruta del directorio del archivo actual
current_dir = os.path.dirname(__file__)

model_ml_asma = os.path.join(current_dir, 'models', 'asma_model.pkl')

with open(model_ml_asma, 'rb') as file:
    model_training = pickle.load(file)

# Ruta principal
@app.get("/")
async def ping():
    return {"message": "API desde FastAPI!!!"}

# Ruta para realizar predicción en el modelo
@app.post("/evaluation")
async def add_evaluation_user(request: Request):
    req_data = await request.json()  # Obtener el cuerpo de la solicitud

    # DATOS PARA USAR CON FLUTTER
    BMI=req_data.get('BMI')
    Wheezing=req_data.get('Wheezing')
    ShortnessOfBreath=req_data.get('ShortnessOfBreath')
    ChestTightness=req_data.get('ChestTightness')
    Coughing=req_data.get('Coughing')
    

    # Convertir datos en DataFrame para la predicción
    params_news_df = pd.DataFrame([[BMI, Wheezing, ShortnessOfBreath, ChestTightness, Coughing]],
                                  columns=["BMI", "Wheezing", "ShortnessOfBreath", 'ChestTightness', 'Coughing'])
    print("PARAMETROS EN SOLICITUD POST")
    print(params_news_df)
    
    # Realizar predicción
    result_model = model_training.predict(params_news_df)
    
    if result_model[0] == 0:
        mensaje = "Su evaluación indica un bajo riesgo de desarrollar crisis asmatica."
    else:
        mensaje = "Su evaluación indica un alto riesgo de desarrollar crisis asmatica."
    
    print(result_model[0])
    return JSONResponse(content={'AsthmaDiagnosis': str(result_model), "message": mensaje})

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8080)

