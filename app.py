from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_efficientnet.onnx")
EMOTIONS   = ["Angry", "Happy", "Neutral", "Sad"]
IDX_4      = [0, 4, 5, 6]
MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Load model saat server nyala
sess   = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
IN_NM  = sess.get_inputs()[0].name
OUT_NM = sess.get_outputs()[0].name
SH     = sess.get_inputs()[0].shape
IMG_SZ = (SH[3], SH[2]) if len(SH) == 4 else (260, 260)
det    = cv2.CascadeClassifier(
             cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print(f"✅ Model ready | input: {IMG_SZ}")


@app.post("/detect-mood")
async def detect_mood(image: UploadFile = File(...)):

    # 1. Decode gambar
    raw = await image.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, detail="Gambar tidak valid")

    # 2. Crop wajah, fallback ke seluruh gambar
    gray  = cv2.createCLAHE(2.0,(8,8)).apply(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    faces = det.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
    if len(faces) > 0:
        x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
        pad = int(0.15*min(w,h)); H,W = img.shape[:2]
        img = img[max(0,y-pad):min(H,y+h+pad),
                  max(0,x-pad):min(W,x+w+pad)]

    # 3. Preprocess → model
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[:,:,0] = cv2.createCLAHE(2.0,(4,4)).apply(ycc[:,:,0])
    rgb = cv2.cvtColor(cv2.cvtColor(ycc,cv2.COLOR_YCrCb2BGR),cv2.COLOR_BGR2RGB)
    rs  = cv2.resize(rgb, IMG_SZ, interpolation=cv2.INTER_LINEAR)
    n   = (rs.astype("float32")/255.0 - MEAN) / STD
    inp = n.transpose(2,0,1)[np.newaxis].astype(np.float32)

    # 4. Inferensi
    logits = sess.run([OUT_NM], {IN_NM: inp})[0][0]
    logits -= logits.max()
    p8  = np.exp(logits) / np.exp(logits).sum()
    p4  = p8[IDX_4]; p4 /= p4.sum() + 1e-8

    best = int(np.argmax(p4))

    # 5. Return JSON
    return {
        "mood"        : EMOTIONS[best],
        "confidence"  : round(float(p4[best]), 4),
        "all_emotions": {e: round(float(p), 4)
                         for e, p in zip(EMOTIONS, p4)},
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
