"""
Routes for the emotion detection application
"""

import os

from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.utils import secure_filename

from app.database import EmotionRecord, User, db
from app.model_loader import EmotionDetector
from app.utils import allowed_file, get_emotion_color, preprocess_image

main = Blueprint("main", __name__)

# Initialize emotion detector
detector = EmotionDetector()


@main.route("/")
def index():
    """Render the main page"""
    return render_template("index.html")


@main.route("/api/detect", methods=["POST"])
def detect_emotion():
    """
    API endpoint to detect emotion from uploaded image
    Expects: multipart/form-data with 'image' file, 'name', and 'email'
    Returns: JSON with emotion predictions
    """
    try:
        # Get user information from form data
        name = request.form.get("name")
        email = request.form.get("email")

        # Validate user information
        if not name or not email:
            return jsonify({"error": "Name and email are required"}), 400

        # Check if image file is present
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]

        # Check if file is selected
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Validate file type
        if not allowed_file(file.filename, current_app.config["ALLOWED_EXTENSIONS"]):
            return (
                jsonify(
                    {
                        "error": "Invalid file type. Allowed types: PNG, JPG, JPEG, GIF, BMP"
                    }
                ),
                400,
            )

        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Preprocess the image
        processed_image = preprocess_image(
            filepath,
            target_size=current_app.config["IMAGE_SIZE"],
            grayscale=current_app.config["GRAYSCALE"],
        )

        # Detect emotion
        predictions = detector.predict(processed_image)
        dominant_emotion = max(predictions, key=predictions.get)

        # Get or create user
        user = User.query.filter_by(email=email).first()
        if not user:
            user = User(name=name, email=email)
            db.session.add(user)
            db.session.flush()  # Get user ID without committing

        # Create emotion record
        emotion_record = EmotionRecord(
            user_id=user.id,
            image_path=f"/static/uploads/{filename}",
            dominant_emotion=dominant_emotion,
            confidence=predictions[dominant_emotion],
            angry=predictions.get("Angry", 0.0),
            disgust=predictions.get("Disgust", 0.0),
            fear=predictions.get("Fear", 0.0),
            happy=predictions.get("Happy", 0.0),
            sad=predictions.get("Sad", 0.0),
            surprise=predictions.get("Surprise", 0.0),
            neutral=predictions.get("Neutral", 0.0),
        )
        db.session.add(emotion_record)
        db.session.commit()

        # Format response
        response = {
            "success": True,
            "predictions": predictions,
            "dominant_emotion": dominant_emotion,
            "image_path": f"/static/uploads/{filename}",
            "user": user.to_dict(),
            "record_id": emotion_record.id,
        }

        return jsonify(response), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@main.route("/api/emotions", methods=["GET"])
def get_emotions():
    """Get list of available emotions"""
    emotions = current_app.config["EMOTIONS"]
    emotion_info = [
        {"name": emotion, "color": get_emotion_color(emotion)} for emotion in emotions
    ]
    return jsonify({"emotions": emotion_info}), 200


@main.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return (
        jsonify(
            {
                "status": "healthy",
                "model_loaded": detector.is_loaded(),
                "version": "1.0.0",
            }
        ),
        200,
    )


@main.route("/api/users", methods=["GET"])
def get_users():
    """Get all users"""
    try:
        users = User.query.all()
        return (
            jsonify({"success": True, "users": [user.to_dict() for user in users]}),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route("/api/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    """Get specific user with their emotion history"""
    try:
        user = User.query.get_or_404(user_id)
        emotion_records = (
            EmotionRecord.query.filter_by(user_id=user_id)
            .order_by(EmotionRecord.analyzed_at.desc())
            .all()
        )

        return (
            jsonify(
                {
                    "success": True,
                    "user": user.to_dict(),
                    "history": [record.to_dict() for record in emotion_records],
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route("/api/users/<string:email>", methods=["GET"])
def get_user_by_email(email):
    """Get user by email with their emotion history"""
    try:
        user = User.query.filter_by(email=email).first_or_404()
        emotion_records = (
            EmotionRecord.query.filter_by(user_id=user.id)
            .order_by(EmotionRecord.analyzed_at.desc())
            .all()
        )

        return (
            jsonify(
                {
                    "success": True,
                    "user": user.to_dict(),
                    "history": [record.to_dict() for record in emotion_records],
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route("/api/history", methods=["GET"])
def get_all_history():
    """Get all emotion detection history"""
    try:
        limit = request.args.get("limit", 50, type=int)
        records = (
            EmotionRecord.query.order_by(EmotionRecord.analyzed_at.desc())
            .limit(limit)
            .all()
        )

        return (
            jsonify(
                {"success": True, "records": [record.to_dict() for record in records]}
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route("/realtime")
def realtime():
    """Render real-time detection page"""
    return render_template("realtime.html")


# Global variable to store latest emotion data
latest_emotion_data = {
    "faces": 0,
    "predictions": {},
    "dominant_emotion": "None",
    "timestamp": 0,
}


@main.route("/api/emotion_data")
def get_emotion_data():
    """Get latest emotion detection data"""
    return jsonify(latest_emotion_data)


@main.route("/api/video_feed")
def video_feed():
    """Video streaming route"""
    import cv2
    import numpy as np
    from flask import Response

    from app.realtime_detection import RealtimeEmotionDetector

    def generate_frames():
        detector = RealtimeEmotionDetector()

        # Try different camera indices
        camera = None
        for index in [0, 1, -1]:
            camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Use DirectShow on Windows
            if camera.isOpened():
                print(f"✓ Camera opened successfully on index {index}")
                break
            camera.release()

        if camera is None or not camera.isOpened():
            print("Error: Could not open webcam")
            # Return error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                error_frame,
                "Camera Not Available",
                (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            ret, buffer = cv2.imencode(".jpg", error_frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            return

        # Set camera properties for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced from 1280
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced from 720
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            import time

            while True:
                success, frame = camera.read()
                if not success:
                    break

                # Resize frame for faster processing
                frame = cv2.resize(frame, (640, 480))

                # Detect faces
                faces = detector.detect_faces(frame)

                # Update global emotion data
                if len(faces) > 0:
                    # Get predictions from first face
                    predictions = detector.process_face(frame, faces[0])
                    dominant_emotion = max(predictions, key=predictions.get)

                    # Update global data
                    global latest_emotion_data
                    latest_emotion_data = {
                        "faces": len(faces),
                        "predictions": predictions,
                        "dominant_emotion": dominant_emotion,
                        "timestamp": time.time(),
                    }

                    # Process all faces for display
                    for face in faces:
                        face_predictions = detector.process_face(frame, face)
                        frame = detector.draw_predictions(frame, face, face_predictions)
                else:
                    # No faces detected
                    latest_emotion_data = {
                        "faces": 0,
                        "predictions": {},
                        "dominant_emotion": "None",
                        "timestamp": time.time(),
                    }

                # Add face count
                cv2.putText(
                    frame,
                    f"Faces: {len(faces)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # Encode frame with compression
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                ret, buffer = cv2.imencode(".jpg", frame, encode_param)
                frame_bytes = buffer.tobytes()

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

        except Exception as e:
            print(f"Error in video feed: {str(e)}")
        finally:
            camera.release()
            print("Camera released")

    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )
