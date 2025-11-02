"""
Database models and initialization
"""

from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    """User model for storing user information and emotion detection results"""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship to emotion records
    emotions = db.relationship(
        "EmotionRecord", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User {self.name} ({self.email})>"

    def to_dict(self):
        """Convert user to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "total_analyses": len(self.emotions),
        }


class EmotionRecord(db.Model):
    """Emotion detection record model"""

    __tablename__ = "emotion_records"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    dominant_emotion = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

    # Store all emotion predictions as JSON-like columns
    angry = db.Column(db.Float, default=0.0)
    disgust = db.Column(db.Float, default=0.0)
    fear = db.Column(db.Float, default=0.0)
    happy = db.Column(db.Float, default=0.0)
    sad = db.Column(db.Float, default=0.0)
    surprise = db.Column(db.Float, default=0.0)
    neutral = db.Column(db.Float, default=0.0)

    analyzed_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<EmotionRecord {self.dominant_emotion} - {self.confidence:.2f}%>"

    def to_dict(self):
        """Convert emotion record to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "image_path": self.image_path,
            "dominant_emotion": self.dominant_emotion,
            "confidence": self.confidence,
            "predictions": {
                "Angry": self.angry,
                "Disgust": self.disgust,
                "Fear": self.fear,
                "Happy": self.happy,
                "Sad": self.sad,
                "Surprise": self.surprise,
                "Neutral": self.neutral,
            },
            "analyzed_at": self.analyzed_at.isoformat(),
        }


def init_db(app):
    """Initialize database with app context"""
    db.init_app(app)

    with app.app_context():
        db.create_all()
        print("✓ Database initialized successfully")
