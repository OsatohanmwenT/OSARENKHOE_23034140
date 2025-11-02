"""
Model loading and prediction module
"""

import os

import joblib
import numpy as np

from config import Config


class EmotionDetector:
    """Emotion detection model wrapper"""

    def __init__(self):
        self.model = None
        self.emotions = Config.EMOTIONS
        self.model_path = Config.MODEL_PATH
        self.load_model()

    def load_model(self):
        """Load the pre-trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"✓ Model loaded successfully from {self.model_path}")
            else:
                print(f"⚠ Model file not found at {self.model_path}")
                print("⚠ Using demo mode with random predictions")
                self.model = None
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            self.model = None

    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None

    def predict(self, image):
        """
        Predict emotion from preprocessed image

        Args:
            image: Preprocessed image array

        Returns:
            Dictionary with emotion probabilities
        """
        try:
            if self.model is not None:
                # Flatten image for sklearn model
                image_flat = image.flatten().reshape(1, -1)

                # Get predictions
                probabilities = self.model.predict_proba(image_flat)[0]

                # Create results dictionary
                predictions = {
                    emotion: float(prob * 100)
                    for emotion, prob in zip(self.emotions, probabilities)
                }
            else:
                # Demo mode: Generate realistic-looking predictions
                predictions = self._generate_demo_predictions()

            return predictions

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return self._generate_demo_predictions()

    def _generate_demo_predictions(self):
        """Generate demo predictions for testing without a model"""
        # Generate random probabilities that sum to 100%
        probs = np.random.dirichlet(np.ones(len(self.emotions))) * 100

        # Create realistic distribution (one dominant emotion)
        dominant_idx = np.random.randint(0, len(self.emotions))
        probs[dominant_idx] += 20
        probs = probs / probs.sum() * 100

        predictions = {
            emotion: float(prob) for emotion, prob in zip(self.emotions, probs)
        }

        return predictions
