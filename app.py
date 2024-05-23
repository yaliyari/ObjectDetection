import io

import cv2
import cvlib
import uvicorn
import numpy as np
import nest_asyncio
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cvlib as cv
from cvlib.object_detection import draw_bbox

# first we need to assign an instance of FastAPI class to the variable app and interact with api using this instance
app = FastAPI(title="Deploying a ML model wuth FastAPI")
# list of availbale models using Enum
class Model(str, Enum):
    yolov4 = "yolov4"
    yolov4tiny = "yolov4_tiny"

# @app.get("/") allows the get method to work for the / endpoint
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs."

# This endpoint handles all the logic necessary for the object detection to work.
# It requires the desired model and the image in which to perform object detection.
@app.post("/predict")
def prediction(model: Model, file: UploadFile = File(...)):

    # 1. Validate Input File
    filename = file.filename
    fileEtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileEtension:
        raise HTTPException(status_code=415, detail="Unsupported file prodided")

    # 2. Transform Wat Image into CV2 image
    # read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())

    # Start the steam from the beginning (position zero)
    image_stream.seek(0)

    # Wrtie the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 3. Run Object Detection Model
    # Run object detection
    bbox, label, conf = cv.detect_common_objects(image, model=model)

    # create image with bounding box and label
    output_image = draw_bbox(image, bbox, label, conf)

    # Save it in a folder within the server
    cv2.imwrite(f'images_uploaded/{filename}', output_image)

    # Stream the Response Back to the Client
    # Open the saves image for reading in binary mode
    file_image = open(f'/images_uploaded/{filename}', mode='rb')

    # Return the image as a stream specifying the media type
    return StreamingResponse(file_image, media_type="image/jpeg")

