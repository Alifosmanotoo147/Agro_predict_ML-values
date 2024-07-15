from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the pre-trained model
model = joblib.load('crop_regressor_model.joblib')

crop_mapping = {
    'rice': 0, 'maize': 1, 'chickpea': 2, 'kidneybeans': 3, 'pigeonpeas': 4,
    'mothbeans': 5, 'mungbean': 6, 'blackgram': 7, 'lentil': 8, 'pomegranate': 9,
    'banana': 10, 'mango': 11, 'grapes': 12, 'watermelon': 13, 'muskmelon': 14,
    'apple': 15, 'orange': 16, 'papaya': 17, 'coconut': 18, 'cotton': 19, 'jute': 20,
    'coffee': 21
}

class CropRequest(BaseModel):
    crop_name: str

@app.post("/predict")
def predict_crop(request: CropRequest):
    crop_name = request.crop_name.lower()
    if crop_name in crop_mapping:
        crop_label = crop_mapping[crop_name]
        prediction = model.predict([[crop_label]])
        return {
            "crop_name": crop_name,
            "N": prediction[0][0],
            "P": prediction[0][1],
            "K": prediction[0][2],
            "temperature": prediction[0][3],
            "humidity": prediction[0][4]
        }
    else:
        return {"error": "Crop name not found in the dataset."}
