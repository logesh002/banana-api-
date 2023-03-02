from fastapi import FastAPI, File, UploadFile
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import requests
from fastapi.middleware.cors import CORSMiddleware


app=FastAPI()


origins = [
    
    "http://localhost:3000",
    "https://aquamarine-valkyrie-4f96dc.netlify.app",
    "https://astonishing-quokka-246291.netlify.app/",
    "https://astonishing-quokka-246291.netlify.app",
    "https://silly-cactus-ed7ff8.netlify.app",
     "https://delicate-alpaca-93e64c.netlify.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL =tf.keras.models.load_model("models/Bananamodel3.h5",compile=False)
MODEL.compile(
    loss = 'categorical_crossentropy',
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ['accuracy']
)
ENDPOINT="http://localhost:8601/v1/models/banana:predict"
CLASS_NAMES =['NotBanana','cordana', 'healthy', 'pestalotiopsis', 'sigatoka']

def read_file_as_image(data)->np.array:
    img_path=Image.open(BytesIO(data))
    img = image.load_img(BytesIO(data), target_size=(224, 224))
    #img = image.load_img(BytesIO(data), target_size=(256, 256))
    img_array=image.img_to_array(img)
    return img_array

@app.get("/ping")
async def ping():
    return "Hello, I am alive"


@app.post("/predict")
async def predict(file: UploadFile =File(...)):
    image = read_file_as_image(await file.read())
    image_batch=np.expand_dims(image,0)
    predictions=MODEL.predict(image_batch)
    index=np.argmax(predictions[0])
    predicted_class=CLASS_NAMES[index]
    confidence=np.max(predictions[0]) 
    return {"prediction":predicted_class , "confidence":float(confidence)}
    

if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)
