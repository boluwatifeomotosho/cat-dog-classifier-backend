from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import logging
from typing import Optional

import numpy as np
from PIL import Image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Application configuration."""
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'cat_dog_model.h5')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def load_trained_model(model_path: str) -> Optional[Model]:
    """Load the pre-trained model from the specified path."""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}. The API will not be able to make predictions.")
        return None
    try:
        model = load_model(model_path)
        logger.info(f"Model '{model_path}' loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess PIL image for prediction."""
    try:
        # Resize image to the model's expected input size (e.g., 64x64)
        img = img.resize((64, 64))
        # Convert to RGB if necessary, as model expects 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Convert to array
        img_array = image.img_to_array(img)
        # Rescale pixel values to [0,1]
        img_array /= 255.0
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def register_routes(app: Flask):
    """Register Flask routes."""

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        model_is_loaded = hasattr(app, 'model') and app.model is not None
        return jsonify({
            'status': 'healthy' if model_is_loaded else 'unhealthy',
            'model_loaded': model_is_loaded
        })

    @app.route('/predict', methods=['POST'])
    def predict():
        """Prediction endpoint."""
        if not hasattr(app, 'model') or app.model is None:
            return jsonify({
                'error': 'Model is not loaded on the server.',
                'message': 'The prediction model is not available on the server.'
            }), 503  # Service Unavailable

        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided.'}), 400

        file = request.files['image']

        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400

        # Check file type
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in app.config['ALLOWED_EXTENSIONS']:
            return jsonify({
                'error': 'Invalid file type.',
                'message': f"Allowed file types are: {', '.join(app.config['ALLOWED_EXTENSIONS'])}"
            }), 415  # Unsupported Media Type

        try:
            # Read and process image
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))

            # Preprocess image
            processed_img = preprocess_image(img)

            # Make prediction
            prediction = app.model.predict(processed_img)
            confidence = float(prediction[0][0])

            # Interpret result (assuming 1 = dog, 0 = cat based on your training)
            if confidence > 0.5:
                predicted_class = 'dog'
                confidence_percentage = confidence * 100
            else:
                predicted_class = 'cat'
                confidence_percentage = (1 - confidence) * 100

            return jsonify({
                'prediction': predicted_class,
                'confidence': round(confidence_percentage, 2)
            })

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({'error': 'Failed to process image and predict.'}), 500
            
    @app.route('/model-info', methods=['GET'])
    def model_info():
        """Get model information."""
        if not hasattr(app, 'model') or app.model is None:
            return jsonify({'error': 'Model is not loaded.'}), 503

        try:
            return jsonify({
                'input_shape': app.model.input_shape,
                'output_shape': app.model.output_shape,
                'total_params': int(app.model.count_params()),
                'classes': ['cat', 'dog']
            })
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return jsonify({'error': 'Could not retrieve model information.'}), 500

# This is the single entry point for the application
def create_app(config_object: Config = Config) -> Flask:
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_object)

    # Configure CORS for production. Use an environment variable for the frontend URL.
    # This allows all origins in development if the variable isn't set.
    frontend_url = os.environ.get('FRONTEND_URL')
    CORS(app, origins=[frontend_url] if frontend_url else "*")

    # Load the model within the application context to ensure it's available for all workers.
    with app.app_context():
        app.model = load_trained_model(app.config['MODEL_PATH'])

    register_routes(app)

    return app

# Create the app instance for WSGI servers like Gunicorn
app = create_app()

if __name__ == '__main__':
    # This block allows running the app directly with `python app.py` for local development.
    port = int(os.environ.get('PORT', 5000))
    # Use debug=False to prevent the reloader from loading the model twice.
    # The reloader is not necessary as the model is loaded once at startup.
    app.run(host='0.0.0.0', port=port, debug=True)
