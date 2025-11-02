"""
Main entry point for the Emotion Detection AI Application
"""

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print("🎭 Emotion Detection AI Application")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("Press CTRL+C to quit")
    print("=" * 60)

    app.run(debug=True, host="0.0.0.0", port=5000)
