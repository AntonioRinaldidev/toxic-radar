from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from detoxify import Detoxify
import torch

app = FastAPI(title="ToxicRadar Classifier")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Detoxify('unbiased', device=device)

class TextRequest(BaseModel):
    text: str

@app.post("/classify")
def classify_text(request: TextRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Il testo non può essere vuoto.")

    try:
        results = model.predict(text)
        results_clean = {k: float(v) for k, v in results.items()}
        return {
            "isSuccess": True,
            "message": "Classification completed.",
            "data": results_clean
        }
    except Exception as e:
        return {
            "isSuccess": False,
            "message": str(e),
            "data": None
        }
