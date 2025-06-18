from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from detoxify import Detoxify
import torch

app = FastAPI(title="ToxicRadar Classifier")

# Usa GPU se disponibile, altrimenti CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Detoxify('unbiased', device=device)

# Definizione dello schema JSON in input
class TextRequest(BaseModel):
    text: str

@app.post("/classify")
def classify_text(request: TextRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Il testo non può essere vuoto.")

    results = model.predict(text)
    results_clean = {k: float(v) for k, v in results.items()}
    return {"labels": results_clean}
