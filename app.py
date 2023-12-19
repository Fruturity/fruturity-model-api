import os

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from google.cloud import storage

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])
app.config["UPLOAD_FOLDER"] = "static/uploads/"

def allowed_file(filename):
    return "." in filename and filename.split(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]

def take_model(bucket_name, model_name):
    storage_client = storage.Client.from_service_account_json("bucket_model.json")
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(model_name)
    local_model_filename = '/fruit_detection2.h5'
    blob.download_to_filename(local_model_filename)
    return load_model(local_model_filename)

# model = load_model(blob, compile=False)

model = take_model('fruit_model_prediction', 'fruit_detection.h5')

@app.route("/")
def index():
    return jsonify({
        "message": "Hi"
    })

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            img = Image.open(image_path).convert("RGB")
            img = img.resize((224,224))
            img_array = np.asarray(img)
            img_array = np.expand_dims(img_array, axis=0)
            normalized_image_array = np.vstack([img_array])

            data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
            data[0] = normalized_image_array

            prediction = model.predict(data, batch_size=10)
            index = np.argmax(prediction[0])

            if index == 0:
                type = "Ripe"
            elif index == 1:
                type = "Unripe"
            else:
                type = "Rotten"


            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Prediction success"
                },
                "data": {
                    "fruit_types_prediction": type
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Client side error"
                },
                "data": None
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None
        }), 405


if __name__ == "__main__":
    app.run()
