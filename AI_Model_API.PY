# ✅ Step 1: Install required packages
!pip install flask flask-cors pyngrok tensorflow pillow

# ✅ Step 2: Import all modules
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image

model_path = "/content/new_model.h5"
model = load_model(model_path)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Step 4: Define class labels
class_names = [
    'Actinic keratoses (akiec)',
    'Basal cell carcinoma (bcc)',
    'Benign keratosis-like lesion (bkl)',
    'Dermatofibroma (df)',
    'Melanoma (mel)',
    'Melanocytic nevi (nv)',
    'Vascular lesion (vasc)'
]

# ✅ Step 5: Preprocessing function
def preprocess_image(base64_str):
    image_data = base64.b64decode(base64_str.split(',')[1])  # remove 'data:image/...;base64,'
    image = Image.open(io.BytesIO(image_data)).resize((224, 224))  # 🔁 adjust if model uses other size
    image = np.array(image) / 255.0  # normalize
    return np.expand_dims(image, axis=0)
# ✅ Step 6: Create Flask API
app = Flask(__name__)
CORS(app)  # allow frontend access

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        image_data = data['image']
        img = preprocess_image(image_data)
        prediction = model.predict(img)
        index = np.argmax(prediction[0])
        confidence = float(prediction[0][index])
        result = {
            "prediction": class_names[index],
            "confidence": confidence
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
!ngrok config add-authtoken 2slQ88H6lcHEdMKBsgEVVeDEDQa_mb7YMiC9HYTuFSnV946g


 #✅ Step 7: Start API and expose via ngrok
public_url = ngrok.connect(5000)
print("🔗 Public URL for API:", public_url)

# Start Flask
app.run(port=5000)