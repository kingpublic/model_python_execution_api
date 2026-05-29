
```markdown
# Mood Bites - Emotion Detection API 🎭🍔

A FastAPI-based backend service for detecting human emotions from facial images. This API acts as the core computer vision engine for the **Mood Bites** project, analyzing user moods to seamlessly integrate with an IoT system for personalized food recommendations.

## 📌 Features
* **Fast & Lightweight:** Built on FastAPI for high performance.
* **Smart Face Cropping:** Utilizes OpenCV's Haar Cascade to detect and isolate faces before passing them to the model.
* **ONNX Runtime:** Uses an optimized EfficientNet model (`emotion_efficientnet.onnx`) for fast CPU inference.
* **Cross-Platform:** Uses dynamic path resolution (`os.path`), making it safe to deploy on both Windows and Linux (VPS) without altering the code.
* **4-Class Emotion Output:** Accurately classifies emotions into `Angry`, `Happy`, `Neutral`, and `Sad`.

## 📁 Project Structure

```text
MODEL_PYTHON_EXECUTION_API/
│
├── app.py                      # Main FastAPI application script
├── emotion_efficientnet.onnx   # Pre-trained ONNX emotion detection model
├── requirements.txt            # List of Python dependencies
├── README.md                   # Project documentation
└── .gitattributes              # Git configurations (e.g., for Git LFS)

```

## 🚀 Installation & Setup

1. **Clone the repository (if applicable):**
```bash
git clone <your-github-repo-url>
cd <your-repository-folder>

```


2. **Install the required dependencies:**
Make sure you have Python installed. It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt

```


*(Ensure `requirements.txt` contains: `fastapi`, `uvicorn`, `python-multipart`, `onnxruntime`, `numpy`, `opencv-python`)*

## 💻 How to Run the Server

Start the API server by running the Python script directly:

```bash
python app.py

```

Alternatively, you can run it using Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

```

You should see a message in the terminal indicating that the model is ready and Uvicorn is running:
`✅ Model ready | input: (260, 260)`
`INFO: Uvicorn running on http://0.0.0.0:8000`

## 📡 API Endpoints

### 1. Health Check

Check if the server is running properly.

* **URL:** `/health`
* **Method:** `GET`
* **Response:**
```json
{
    "status": "ok"
}

```



### 2. Detect Mood

Upload an image to detect the dominant emotion.

* **URL:** `/detect-mood`
* **Method:** `POST`
* **Content-Type:** `multipart/form-data`
* **Body:** `image` (File - The image containing a face)
* **Success Response:**
```json
{
    "mood": "Happy",
    "confidence": 0.9421,
    "all_emotions": {
        "Angry": 0.0123,
        "Happy": 0.9421,
        "Neutral": 0.0341,
        "Sad": 0.0115
    }
}

```



## 🧪 Testing the API

You can easily test the API directly from your browser without any frontend!

1. Run the server.
2. Open your web browser and navigate to: **http://localhost:8000/docs**
3. Use the built-in **Swagger UI** to upload images and see the JSON response instantly.

```

```