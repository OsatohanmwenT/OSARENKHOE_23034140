"""
Flask application factory
"""

import os

from flask import Flask

from config import config


def create_app(config_name="default"):
    """Create and configure the Flask application"""
    app = Flask(__name__, template_folder="../templates", static_folder="../static")

    # Load configuration
    app.config.from_object(config[config_name])

    # Ensure upload folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Initialize database
    from app.database import init_db
    init_db(app)

    # Register blueprints
    from app.routes import main

    app.register_blueprint(main)

    return app
