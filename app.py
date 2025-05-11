import pickle
import os
import pandas as pd
import numpy as np
import time
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
    BMI=req_data.get('questionIMC')
    Wheezing=req_data.get('questionWheezing')
    ShortnessOfBreath=req_data.get('questionShortnessOfBreath')
    ChestTightness=req_data.get('questionChestTightness')
    Coughing=req_data.get('questionCoughing')
    

    # Convertir datos en DataFrame para la predicción
    params_news_df = pd.DataFrame([[BMI, Wheezing, ShortnessOfBreath, ChestTightness, Coughing]],
                                  columns=["BMI", "Wheezing", "ShortnessOfBreath", 'ChestTightness', 'Coughing'])
    print("PARAMETROS EN SOLICITUD POST")
    print(params_news_df)
    
    # Realizar predicción
    start_time = time.time()
    result_model = model_training.predict(params_news_df)
    end_time = time.time()

    prediction_time = round((end_time - start_time) * 1000, 2)
    
    if result_model[0] == 0:
        mensaje = "Su evaluación indica una baja probabilidad de crisis asmatica."
    else:
        mensaje = "Su evaluación indica una alta probabilidad de crisis asmatica."
    
    print(result_model[0])
    print("Tiempo de predicción:", prediction_time, "ms")
    return JSONResponse(
        content={'AsthmaDiagnosis': 
                 str(result_model), 
                 "message": mensaje, 
                 'prediction_time_ms': prediction_time 
                 })

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8080)

