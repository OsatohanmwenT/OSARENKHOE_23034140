"""
Configuration settings for the application
"""

import os


class Config:
    """Base configuration"""

    SECRET_KEY = (
        os.environ.get("SECRET_KEY") or "your-secret-key-here-change-in-production"
    )

    # Database settings
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get("DATABASE_URL") or "sqlite:///emotion_detection.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload settings
    UPLOAD_FOLDER = "static/uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}

    # Model settings
    MODEL_PATH = "models/emotion_model.pkl"
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    # Image processing settings
    IMAGE_SIZE = (48, 48)
    GRAYSCALE = True


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""

    DEBUG = False


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
