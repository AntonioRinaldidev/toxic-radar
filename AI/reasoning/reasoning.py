from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from logic import apply_reasoning

app = FastAPI(title="ToxicRadar Reasoning Module")

class LabelRequest(BaseModel):
    labels: Dict[str, float]

@app.post("/reason")
def reason_labels(data: LabelRequest):
    try:
        result = apply_reasoning(data.labels)
        return result  # contiene adjusted_labels + explanations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
