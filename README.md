# Advanced Facial Recognition System

An advanced, secure, and robust Facial Recognition System built using Python, OpenCV, and the face_recognition library. This project provides accurate facial recognition, advanced head-pose detection, and an intuitive user interface designed for comfortable use.

## Features
- **Face Registration**: Register multiple angles for improved accuracy.
- **Real-Time Facial Recognition**: Fast, reliable, and stable facial recognition.
- **Head Pose Estimation**: Accurate detection of yaw, pitch, and roll.
- **Modern GUI**: Clean, modern, and intuitive user interface.
- **Encodings Management**: Efficient storage and retrieval of facial encodings.

## Installation

### Requirements
- Python 3.8 or higher
- OpenCV
- face_recognition
- NumPy
- SciPy

### Step-by-step installation:
```bash
git clone https://github.com/your_username/facial-recognition-system.git
cd facial-recognition-system

# Recommended virtual environment
python3 -m venv env
source env/bin/activate

pip install -r requirements.txt
```

### Requirements File (`requirements.txt`)
```
opencv-python
face_recognition
numpy
scipy
```

## Usage

### Face Registration
To register a new face:
```bash
python face_registration.py
```

Follow the on-screen prompts to capture multiple facial angles.

### Face Recognition
Run the facial recognition script:
```bash
python facial_recognizer.py
```

Recognition starts instantly and displays the recognized user's name.

### Controls
- Press `q` to exit any of the running scripts.

## Project Structure
```
.
├── encodings             # Saved facial encodings
├── camera.py             # Camera utility class
├── face_registration.py  # Script to register new faces
├── facial_recognizer.py  # Script to recognize faces in real-time
├── utils.py              # Utility functions for registration & recognition
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation (this file)
```

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
MIT License

---

Made with ❤️ by [Your Name]

