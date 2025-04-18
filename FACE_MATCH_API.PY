!pip install deepface
!pip install flask flask-cors supabase pyngrok

from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2
import base64
import numpy as np
import requests
from io import BytesIO
from supabase import create_client, Client
from PIL import Image
from pyngrok import ngrok

# ✅ Supabase
SUPABASE_URL = "https://csluiwfzgtxuodfffinf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNzbHVpd2Z6Z3R4dW9kZmZmaW5mIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQzNTMwODEsImV4cCI6MjA1OTkyOTA4MX0.XCGYf5q7QD57K8-wf0Pm8BzjdsT2sIWQSMQbAbtn5ss"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ Flask app
app = Flask(__name__)
CORS(app)

# ✅ Helper: base64 to OpenCV image
def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# ✅ Load known faces from Supabase
def load_known_faces():
    patients = supabase.table("patients").select("id, name, face_image").execute()
    known_faces = []

    for p in patients.data:
        try:
            response = requests.get(p['face_url'])
            img = np.array(Image.open(BytesIO(response.content)))
            known_faces.append({'id': p['id'], 'name': p['name'], 'img': img})
        except:
            continue
    return known_faces

known_faces = load_known_faces()

# ✅ Face Recognition API
@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'status': 'error', 'message': 'No image received'}), 400

    input_img = decode_image(data['image'])

    for face in known_faces:
        try:
            result = DeepFace.verify(input_img, face['img'], enforce_detection=False)
            if result['verified']:
                patient = supabase.table("patients").select("*").eq("id", face['id']).single().execute().data
                return jsonify({'status': 'success', 'patient': patient}), 200
        except Exception as e:
            continue

    return jsonify({'status': 'not_found'}), 404

!ngrok config add-authtoken 2vLZHHTR2ram6I4RKDg1ZyqLLVG_7yPjmPb8BaVrCT23g46vD


 #✅ Step 7: Start API and expose via ngrok
public_url = ngrok.connect(5000)
print("🔗 Public URL for API:", public_url)

# Start Flask
app.run(port=5000)