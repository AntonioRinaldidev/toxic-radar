from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

CLASSIFIER_URL = "http://localhost:8001/classify"
REASONING_URL = "http://localhost:8002/reason"

class AnalyzeRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(req: AnalyzeRequest):
    try:
        # Step 1: Classify
        clf_response = requests.post(CLASSIFIER_URL, json={"text": req.text})
        clf_response.raise_for_status()
        clf_data = clf_response.json()

        if not clf_data.get("isSuccess"):
            raise Exception("Classifier error: " + clf_data.get("message", ""))

        raw_labels = clf_data["data"]

        # Step 2: Apply reasoning
        rsn_response = requests.post(REASONING_URL, json={"labels": raw_labels})
        rsn_response.raise_for_status()
        rsn_data = rsn_response.json()

        return {
            "isSuccess": True,
            "message": "Full analysis completed.",
            "data": {
                "text": req.text,
                "raw_labels": raw_labels,
                "adjusted_labels": rsn_data["adjusted_labels"],
                "explanations": rsn_data["explanations"]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
