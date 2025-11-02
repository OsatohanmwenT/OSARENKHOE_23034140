"""
Utility functions for image processing and validation
"""

import cv2
from PIL import Image


def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def preprocess_image(image_path, target_size=(48, 48), grayscale=True):
    """
    Preprocess image for emotion detection

    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
        grayscale: Whether to convert to grayscale

    Returns:
        Preprocessed image array
    """
    try:
        # Read image
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError("Failed to load image")

        # Convert to grayscale if needed
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to target size
        img = cv2.resize(img, target_size)

        # Normalize pixel values
        img = img.astype("float32") / 255.0

        return img

    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")


def get_emotion_color(emotion):
    """Get color code for emotion"""
    colors = {
        "Happy": "#FFD700",  # Gold
        "Sad": "#4169E1",  # Royal Blue
        "Angry": "#DC143C",  # Crimson
        "Surprise": "#FF69B4",  # Hot Pink
        "Fear": "#9370DB",  # Medium Purple
        "Disgust": "#32CD32",  # Lime Green
        "Neutral": "#808080",  # Gray
    }
    return colors.get(emotion, "#000000")


def validate_image_dimensions(image_path, max_width=4096, max_height=4096):
    """Validate image dimensions"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            if width > max_width or height > max_height:
                return (
                    False,
                    f"Image too large: {width}x{height}. Max: {max_width}x{max_height}",
                )

            if width < 48 or height < 48:
                return False, f"Image too small: {width}x{height}. Min: 48x48"

            return True, "Valid"

    except Exception as e:
        return False, f"Error validating image: {str(e)}"
