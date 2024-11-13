from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
import io
import tensorflow as tf
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and define class names
class_names = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast',
               'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']
model_path = "/Users/gobindarora/Paddy_Disease/backend/model/final_paddy_disease_classifier.keras"
model = load_model(model_path)

print("Model exists:", os.path.exists(model_path))

# Function to prepare image for prediction
def prepare_image(file):
    try:
        # Convert image to RGB, resize, and preprocess to match the model's input format
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((224, 224))
        img_array = img_to_array(img)  # Normalize the image to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    
    except Exception as e:
        print(f"Error preparing image: {str(e)}")
        raise e

# Home route to render index.html
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        print("No file part in request")
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        print("No file selected")
        return jsonify({"error": "No file selected"}), 400

    try:
        # Log file information
        print(f"Received file: {file.filename}")

        # Prepare image for prediction
        img_array = prepare_image(file)
        print("Image prepared successfully.")

        # Perform prediction and calculate prediction confidence
        pred_probs = model.predict(img_array)
        pred_class_idx = np.argmax(pred_probs)
        pred_class = class_names[pred_class_idx]
        confidence = pred_probs[0][pred_class_idx]  # Get confidence score for the predicted class

        # Log prediction result
        print(f"Prediction: {pred_class}, Confidence: {confidence}")
        return jsonify({"predicted_class": pred_class, "confidence": float(confidence)})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)
