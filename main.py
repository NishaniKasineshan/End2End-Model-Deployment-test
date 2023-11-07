from fastapi import FastAPI,File,UploadFile
import uvicorn
import io
from PIL import Image
import numpy as np
import tensorflow as tf

app=FastAPI()

Model=tf.keras.models.load_model("model.h5")#model is uploaded

class_names=["cls1","cls2","cls3"]#classes to be predicted

#Function defined to convert bytes to image 
def convert_bytes_to_image(data):
    image=np.array(Image.open(io.BytesIO(data)))
    return image

#define post method for prediction
@app.post("/predict")
async def predict_image(file:UploadFile=File(...)):
    # bytes=await file.read()#need to convert bytes to image
    image=convert_bytes_to_image(await file.read())
    #since predict expect batch of images and we test for only one image we should convert 1D to 2D
    image=np.expand_dims(image,axis=0)
    
    predictions=Model.predict(image)
    #returns an array where the index 0 gives the predictions
    prediction_class=class_names[np.argmax(predictions[0])]#in order to predict the class we take max from the prediction list
    pred_confidence=np.max(predictions[0])

    return {
        'class':prediction_class,
        'confidence':float(pred_confidence)
    }
    
if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)
