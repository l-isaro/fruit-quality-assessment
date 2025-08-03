from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()

# Load model once
model = tf.keras.models.load_model("models/fruit_model.h5")

# Define class labels
class_labels = [
    'freshapples', 'freshbanana', 'freshcucumber', 'freshokra',
    'freshoranges', 'freshpotato', 'freshtomato',
    'rottenapples', 'rottenbanana', 'rottencucumber', 'rottenokra',
    'rottenoranges', 'rottenpotato', 'rottentomato'
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        pred_idx = int(np.argmax(predictions))
        result = {
            "prediction": class_labels[pred_idx],
            "confidence": float(predictions[pred_idx])
        }
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
