# 🎭 AI Emotion Detection Application

A modern web application that uses machine learning to detect emotions from facial images. Built with Flask, scikit-learn, and a beautiful HTML/CSS/JavaScript frontend.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-red.svg)

## ✨ Features

- 📸 **Image Upload**: Drag-and-drop or click to upload images
- 🧠 **AI-Powered Detection**: Machine learning model detects 7 emotions
- 📊 **Visual Results**: Beautiful charts showing emotion probabilities
- 👤 **User Profiles**: Save user information with each analysis
- 💾 **SQLite Database**: Store user data and emotion history
- 📈 **History Tracking**: View past emotion detections
- 🎨 **Modern UI**: Responsive, dark-themed interface
- ⚡ **Real-time Processing**: Fast emotion detection
- 🔒 **Secure**: File validation and size limits

## 🎯 Detected Emotions

- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😲 Surprise
- 😨 Fear
- 🤢 Disgust
- 😐 Neutral

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**
```bash
cd c:\Users\osare\PycharmProjects\OSARENKHOE_23CG034140
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python run.py
```

4. **Open your browser**
```
http://localhost:5000
```

## 📁 Project Structure

```
OSARENKHOE_23CG034140/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── routes.py            # API endpoints
│   ├── model_loader.py      # ML model loader
│   └── utils.py             # Utility functions
├── static/
│   ├── css/
│   │   └── style.css        # Stylesheet
│   ├── js/
│   │   └── app.js           # Frontend logic
│   └── uploads/             # Uploaded images
├── templates/
│   └── index.html           # Main page
├── models/
│   └── emotion_model.pkl    # Trained model
├── notebooks/
│   └── train_model.ipynb    # Training notebook
├── config.py                # Configuration
├── requirements.txt         # Dependencies
├── run.py                   # Entry point
└── README.md               # Documentation
```

## 🔧 Configuration

Edit `config.py` to customize:

- Upload folder location
- Maximum file size
- Allowed file extensions
- Model path
- Image processing settings

## 🎓 Training Your Own Model

1. **Open the Jupyter notebook**
```bash
jupyter notebook notebooks/train_model.ipynb
```

2. **Follow the steps**:
   - Load your dataset (FER2013 recommended)
   - Preprocess images
   - Train the model
   - Save the trained model

3. **Replace the model**:
   - Place your trained model in `models/emotion_model.pkl`

## 💾 Database

The application uses SQLite to store:
- **User Information**: Name and email
- **Emotion Records**: All detected emotions with timestamps
- **Analysis History**: Complete history for each user

View database contents:
```bash
python view_database.py
```

Database file: `emotion_detection.db` (created automatically)

## 🌐 API Endpoints

### `POST /api/detect`
Upload an image and get emotion predictions.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `image` file, `name` string, `email` string

**Response:**
```json
{
  "success": true,
  "predictions": {
    "Happy": 85.3,
    "Sad": 5.2,
    "Angry": 3.1,
    "Surprise": 2.8,
    "Fear": 1.9,
    "Disgust": 1.2,
    "Neutral": 0.5
  },
  "dominant_emotion": "Happy",
  "image_path": "/static/uploads/image.jpg"
}
```

### `GET /api/emotions`
Get list of available emotions with colors.

### `GET /api/health`
Check application health status.

### `GET /api/users`
Get all users.

### `GET /api/users/<user_id>`
Get specific user with emotion history.

### `GET /api/users/<email>`
Get user by email with emotion history.

### `GET /api/history?limit=50`
Get all emotion detection history (default limit: 50).

## 🎨 Customization

### Frontend
- Modify `static/css/style.css` for styling
- Update `static/js/app.js` for behavior
- Edit `templates/index.html` for structure

### Backend
- Add new routes in `app/routes.py`
- Modify image processing in `app/utils.py`
- Update model loading in `app/model_loader.py`

## 📊 Performance

The application includes:
- ✅ Input validation
- ✅ Error handling
- ✅ File size limits (16MB)
- ✅ Image preprocessing
- ✅ Efficient model loading
- ✅ Demo mode (if model not available)

## 🐛 Troubleshooting

### Model not found
- The app runs in demo mode if no model is found
- Train a model using the notebook or download a pre-trained one

### Upload errors
- Check file size (max 16MB)
- Verify file type (PNG, JPG, JPEG, GIF, BMP)
- Ensure `static/uploads/` directory exists

### Port already in use
Change the port in `run.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

## 🔮 Future Enhancements

- [ ] Real-time webcam emotion detection
- [ ] Batch processing for multiple images
- [ ] Emotion history and statistics
- [ ] User authentication
- [ ] Model fine-tuning interface
- [ ] Docker containerization
- [ ] REST API documentation
- [ ] Mobile app integration

## 📝 Dataset Recommendation

For training, we recommend:
- **FER2013**: 35,887 grayscale images (48x48 pixels)
- **CK+**: Extended Cohn-Kanade Dataset
- **AffectNet**: Large-scale facial expression database

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📄 License

This project is for educational purposes.

## 👤 Author

**OSARENKHOE**  
Student ID: 23CG034140

## 🙏 Acknowledgments

- Flask framework
- scikit-learn library
- OpenCV community
- FER2013 dataset creators

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Happy Emotion Detection! 🎭**
