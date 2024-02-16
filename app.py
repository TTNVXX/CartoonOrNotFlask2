from flask import Flask, request, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure file uploads
app.config['UPLOADED_IMAGES_DEST'] = 'static/uploads'  # Ensure directory exists
MAX_FILE_SIZE = 1024 * 1024 * 5  # Set your desired file size limit in bytes (5 MB in this example)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])  # Add allowed file types
images = UploadSet('images', IMAGES)
configure_uploads(app, images)

# Load the CartoonOrNot model
model = tf.keras.models.load_model('CartoonOrNot.h5')

@app.route("/")
def hello():
    return "Cartoon or Not: Image Prediction"

@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        # File handling and validation
        if 'file' not in request.files:
            return "No file found", 400

        image = request.files['file']
        if image.filename == '':
            return "No selected file", 400

        if image.content_length > MAX_FILE_SIZE:
            return "File size exceeds limit", 400

        # Validate file type using allowed extensions
        if '.' not in image.filename or image.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
            return "Invalid file type", 400

        # Save the uploaded image
        filename = secure_filename(image.filename)
        images.save(image, name=filename)

        # Process and predict using saved image
        img_path = os.path.join(app.config['UPLOADED_IMAGES_DEST'], filename)
        image = Image.open(img_path)
        image = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)

        # Perform prediction
        prediction = model.predict(img_array)[0]
        is_cartoon = bool(prediction[0] >= 0.5)

        # Return informative response with prediction result
        return jsonify({
            "prediction": "Cartoon" if is_cartoon else "Not Cartoon",
            "probability": float(prediction[0]) if is_cartoon else float(1 - prediction[0])
        })

    except Exception as e:
        # Log error and return appropriate response
        app.logger.error(f"Error predicting image: {str(e)}")
        return "Internal Server Error: An error occurred while processing the image", 500

if __name__ == "__main__":
    app.run(debug=True)
