import os
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf

# Import your prediction function from your prediction module.
from prediction import predict_eeg_recording

app = FastAPI(title="EEG Epilepsy Prediction API")

# Load the trained model once at startup.
model = tf.keras.models.load_model('model1_2dcnn.h5')

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
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accepts an EEG EDF file, processes it using the preprocessing and spectrogram conversion functions,
    and returns an aggregated prediction (epilepsy or no epilepsy) along with the mean probability.
    """
    # Save the uploaded file temporarily.
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving temporary file")
    
    # Use your prediction function to process the file and obtain an aggregated prediction.
    try:
        pred_label, mean_prob = predict_eeg_recording(tmp_path, model, threshold=0.5)
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    
    os.remove(tmp_path)
    
    response = {
        "prediction": "epilepsy" if pred_label == 1 else "no epilepsy",
        "mean_probability": float(mean_prob)
    }
    return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
