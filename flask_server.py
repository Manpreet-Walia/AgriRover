from flask import Flask, Response, jsonify
import cv2
import time
from supabase import create_client
import os

app = Flask(_name_)  # Fixed underscore syntax

# Supabase credentials (add your Supabase URL and API key here)
SUPABASE_URL = "https://wrktsadpbrocmuyuztxl.supabase.co"  # Replace with your Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Indya3RzYWRwYnJvY211eXV6dHhsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDEzNjcwMzMsImV4cCI6MjA1Njk0MzAzM30.EDH66keNiSQB6wvsYJ-STKPQgIBRNj9QG7sAH0Xvm3s"  # Replace with your Supabase API Key

# Create the Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize video capture
camera = cv2.VideoCapture(0)  # Use 0 for USB camera

# Function to generate video frames
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# API to stream video
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoint to capture image and upload to Supabase
@app.route('/capture_image')
def capture_image():
    try:
        # Capture image from the camera
        ret, frame = camera.read()
        if ret:
            # Convert the frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = buffer.tobytes()
            
            # Generate a filename based on timestamp
            filename = f"image_{int(time.time())}.jpg"
            
            # Upload the image to Supabase Storage
            bucket_name = "img"  # Replace with your bucket name
            bucket = supabase.storage.from_(bucket_name)
            
            # Upload the image to the bucket
            response = bucket.upload(filename, image_data, upsert=True)
            
            if response.status_code == 200:
                print(f"Image uploaded successfully: {filename}")
                return jsonify({"status": "success", "filename": filename}), 200
            else:
                print(f"Error uploading image: {response.status_code}")
                return jsonify({"status": "error", "message": "Failed to upload image"}), 500
        else:
            return jsonify({"status": "error", "message": "Failed to capture image"}), 500
    except Exception as e:
        print(f"Error capturing image: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if _name_ == '_main_':  # Fixed underscore syntax
    app.run(host='0.0.0.0', port=5000, debug=True)  # Added debug=True for easier development