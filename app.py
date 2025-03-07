import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import base64
import io
from PIL import Image
from tensorflow.keras.layers import Layer

app = Flask(__name__)

# Define the custom GaussianNoiseLayer
class GaussianNoiseLayer(Layer):
    def __init__(self, stddev=0.1, **kwargs):
        super(GaussianNoiseLayer, self).__init__(**kwargs)
        self.stddev = stddev
        
    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev, dtype=inputs.dtype)
            return inputs + noise
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev})
        return config

# Load the trained .h5 model with custom objects
MODEL_PATH = "model.h5"  # Ensure this file is in the same directory
model = load_model(MODEL_PATH, custom_objects={"GaussianNoiseLayer": GaussianNoiseLayer})

# Define class labels
CLASS_LABELS = ["Organic", "Inorganic Recyclable", "Inorganic Non-Recyclable"]

# Function to preprocess the image
def preprocess_image(img):
    """Preprocesses the input image: denoising, resizing, and normalizing."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)  # Apply denoising
    img = cv2.resize(img, (224, 224))  # Resize to match the model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model input
    return img

# Route for the homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle image classification
@app.route("/classify", methods=["POST"])
def classify_image():
    try:
        # Read image data from the request
        data = request.get_json()
        image_data = data["image"]

        # Convert base64 string to image
        image_bytes = base64.b64decode(image_data.split(",")[1])
        img = Image.open(io.BytesIO(image_bytes))
        img = np.array(img)

        # Preprocess the image
        processed_img = preprocess_image(img)

        # Make prediction
        predictions = model.predict(processed_img)[0]

        # Get the predicted class
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_LABELS[predicted_class_idx]
        confidence = predictions[predicted_class_idx]

        # Prepare JSON response
        response = {
            "category": predicted_class,
            "confidence": float(confidence),
            "predictions": {CLASS_LABELS[i]: float(predictions[i]) for i in range(len(CLASS_LABELS))}
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
