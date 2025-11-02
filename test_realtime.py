"""
Standalone script for real-time emotion detection
Run this script to test real-time detection without the web server
"""

import os
import sys

import cv2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.realtime_detection import RealtimeEmotionDetector


def main():
    print("=" * 60)
    print("🎥 Real-time Emotion Detection")
    print("=" * 60)
    print("Starting webcam...")
    print("Press 'Q' or 'ESC' to quit")
    print("=" * 60 + "\n")

    # Initialize detector
    detector = RealtimeEmotionDetector()

    # Open webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("❌ Error: Could not open webcam")
        print(
            "Make sure your camera is connected and not in use by another application"
        )
        return

    print("✓ Webcam opened successfully\n")

    # Set camera properties for better performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)

    try:
        while True:
            # Read frame
            success, frame = camera.read()

            if not success:
                print("❌ Error: Could not read frame from webcam")
                break

            # Detect faces
            faces = detector.detect_faces(frame)

            # Process each face
            for face in faces:
                predictions = detector.process_face(frame, face)
                frame = detector.draw_predictions(frame, face, predictions)

            # Add face count
            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Add instructions
            cv2.putText(
                frame,
                "Press 'Q' or 'ESC' to quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Display frame
            cv2.imshow("Real-time Emotion Detection", frame)

            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q") or key == 27:  # Q or ESC
                break

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        print("\n✓ Camera released")
        print("=" * 60)
        print("Goodbye! 👋")
        print("=" * 60)


if __name__ == "__main__":
    main()
