"""
Real-time emotion detection from webcam
"""

import cv2

from app.model_loader import EmotionDetector


class RealtimeEmotionDetector:
    """Real-time emotion detection using webcam"""

    def __init__(self):
        self.detector = EmotionDetector()
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            print("⚠ Warning: Could not load face cascade classifier")

    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return faces

    def process_face(self, frame, face_coords):
        """Process a single face and return emotion predictions"""
        x, y, w, h = face_coords

        # Extract face region
        face_roi = frame[y : y + h, x : x + w]

        # Convert to grayscale
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Resize to model input size
        face_resized = cv2.resize(face_gray, (48, 48))

        # Normalize
        face_normalized = face_resized.astype("float32") / 255.0

        # Predict emotion
        predictions = self.detector.predict(face_normalized)

        return predictions

    def get_emotion_color(self, emotion):
        """Get color for emotion (BGR format for OpenCV)"""
        colors = {
            "Happy": (0, 215, 255),  # Gold
            "Sad": (225, 105, 65),  # Royal Blue
            "Angry": (60, 20, 220),  # Crimson
            "Surprise": (180, 105, 255),  # Hot Pink
            "Fear": (219, 112, 147),  # Medium Purple
            "Disgust": (50, 205, 50),  # Lime Green
            "Neutral": (128, 128, 128),  # Gray
        }
        return colors.get(emotion, (255, 255, 255))

    def draw_predictions(self, frame, face_coords, predictions):
        """Draw bounding box and predictions on frame"""
        x, y, w, h = face_coords

        # Get dominant emotion
        dominant_emotion = max(predictions, key=predictions.get)
        confidence = predictions[dominant_emotion]

        # Get color for emotion
        color = self.get_emotion_color(dominant_emotion)

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Prepare text
        text = f"{dominant_emotion}: {confidence:.1f}%"

        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw background rectangle for text
        cv2.rectangle(
            frame, (x, y - text_height - 10), (x + text_width + 10, y), color, -1
        )

        # Draw text
        cv2.putText(frame, text, (x + 5, y - 5), font, font_scale, (0, 0, 0), thickness)

        # Draw emotion bars on the side
        bar_x = x + w + 10
        bar_y = y
        bar_width = 120
        bar_height = 15

        # Sort emotions by confidence
        sorted_emotions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]

        for i, (emotion, prob) in enumerate(sorted_emotions):
            # Draw background bar
            cv2.rectangle(
                frame,
                (bar_x, bar_y + i * (bar_height + 5)),
                (bar_x + bar_width, bar_y + i * (bar_height + 5) + bar_height),
                (50, 50, 50),
                -1,
            )

            # Draw filled bar
            filled_width = int((prob / 100) * bar_width)
            emotion_color = self.get_emotion_color(emotion)
            cv2.rectangle(
                frame,
                (bar_x, bar_y + i * (bar_height + 5)),
                (bar_x + filled_width, bar_y + i * (bar_height + 5) + bar_height),
                emotion_color,
                -1,
            )

            # Draw emotion label
            label = f"{emotion}: {prob:.0f}%"
            cv2.putText(
                frame,
                label,
                (bar_x + 5, bar_y + i * (bar_height + 5) + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        return frame
