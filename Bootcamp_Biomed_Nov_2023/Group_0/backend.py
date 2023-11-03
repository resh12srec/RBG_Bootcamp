# To install :
# pip install uvicorn
# pip install fastapi
# pip install python-multipart
# pip install Pillow

# To run: uvicorn service:app --host 0.0.0.0 --port 8000

import pickle
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

# Load the trained model from the pickle file
with open('mnist_rf_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

def process_image(image):
    # Open the uploaded image using Pillow (PIL)
    img = Image.open(image)
    
    # Resize and convert to grayscale (if not already)
    img = img.resize((28, 28)).convert('L')
    
    # Convert to a NumPy array
    img_array = np.array(img).flatten()  # Flatten to match the model input shape
    
    # Standardize the data
    img_array = (img_array - np.mean(img_array)) / np.std(img_array)
    
    return img_array

@app.post("/predict/")
async def predict_digit(file: UploadFile):
    try:
        img_array = process_image(BytesIO(await file.read()))
        prediction = clf.predict([img_array])[0]
        return JSONResponse(content={"class": int(prediction)}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
