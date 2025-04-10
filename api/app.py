import os
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Import your prediction functions
from prediction import predict_eeg_recording, predict_ensemble_eeg_recording

app = FastAPI(title="EEG Epilepsy Prediction API")

@app.get("/", tags=["Introduction Endpoints"])
async def index():
    """
    Simply returns a welcome message!
    """
    message = (
        "Hello world! Welcome to the EEG Epilepsy Prediction API. "
        "Submit an EEG recording EDF file to the `/predict` endpoint to receive a prediction."
    )
    return message

@app.post("/predict", tags=["Machine Learning"])
async def predict_endpoint(
    file: UploadFile = File(...),
    model_choice: str = "2DCNN",
    ensemble_method: str = None
):
    """
    
    Query parameters:
      - model_choice: Choose one model among "2DCNN", "EEGNet", "EpilepsyNet", or "ensemble".
      - ensemble_method: (Optional, required if model_choice is "ensemble") 
                           The ensemble method to use ("average" or "voting").

    """
    print("Saving uploaded file as temporary file...")
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving temporary file")
    
    print("Performing prediction using model_choice =", model_choice)
    try:
        if model_choice.lower() == "ensemble":
            if ensemble_method is None:
                raise HTTPException(status_code=400, detail="ensemble_method must be specified when using ensemble model_choice")
            pred_label, mean_prob = predict_ensemble_eeg_recording(tmp_path, ensemble_method=ensemble_method, threshold=0.5)
        else:
            pred_label, mean_prob = predict_eeg_recording(tmp_path, model_name=model_choice, threshold=0.5)
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    
    os.remove(tmp_path)
    
    
    response = {
        "prediction": "epilepsy" if pred_label == 1 else "no epilepsy",
        "confidence": mean_prob
    }
    print("Prediction complete, returning response...")
    return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
