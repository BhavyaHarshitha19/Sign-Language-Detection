"""
ASL Interpretation API
======================
Real-time American Sign Language recognition and translation system.

Endpoints:
  GET  /                     → Serve frontend
  GET  /health               → Health check
  POST /interpret/frame      → Process webcam frame (returns ASL token)
  POST /interpret/finalize   → Convert token sequence to English sentence
  POST /interpret/text       → Direct token interpretation (testing)
"""

import base64
import time
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app.real_asl_recognizer import RealASLRecognizer
from app.simple_interpreter import tokens_to_sentence  # Simple letter display

app = FastAPI(
    title="ASL Recognition API",
    description="Real-time American Sign Language recognition and translation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize real ASL recognizer with MediaPipe hand detection
recognizer = RealASLRecognizer()

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ── Request/Response Models ───────────────────────────────────────────────
class FrameRequest(BaseModel):
    image: str  # base64 data URL

class FrameResponse(BaseModel):
    token: Optional[str] = None
    hand_detected: bool = False

class FinalizeRequest(BaseModel):
    tokens: List[str]
    start_time_ms: Optional[float] = None

class InterpretResponse(BaseModel):
    tokens: List[str]
    sentence: str

class TextRequest(BaseModel):
    tokens: List[str]


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

@app.get("/health")
def health_check():
    """Health check endpoint with model status."""
    model_info = recognizer.get_model_info()
    return {
        "status": "ok", 
        "model_loaded": model_info.get("loaded", False),
        "model_classes": model_info.get("n_classes", 0)
    }


@app.post("/interpret/frame", response_model=FrameResponse)
def interpret_frame(request: FrameRequest):
    """Process a single webcam frame and return ASL token prediction."""
    try:
        # Decode base64 image
        b64_data = request.image.split(",")[-1]
        img_bytes = base64.b64decode(b64_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(400, "Could not decode image")
        
        # Process frame with ASL recognizer
        token, _, hand_detected = recognizer.process_frame(frame)
        
        return FrameResponse(
            token=token,
            hand_detected=hand_detected
        )
        
    except Exception as e:
        raise HTTPException(400, f"Frame processing failed: {str(e)}")


@app.post("/interpret/finalize", response_model=InterpretResponse)
def interpret_finalize(request: FinalizeRequest):
    """Convert accumulated ASL tokens to fluent English sentence."""
    if not request.tokens:
        raise HTTPException(400, "No tokens to interpret")
    
    try:
        sentence = tokens_to_sentence(request.tokens)
        return InterpretResponse(
            tokens=request.tokens,
            sentence=sentence
        )
    except Exception as e:
        raise HTTPException(500, f"Interpretation failed: {str(e)}")


@app.post("/interpret/text", response_model=InterpretResponse)
def interpret_text(request: TextRequest):
    """Direct token interpretation for testing."""
    if not request.tokens:
        raise HTTPException(400, "No tokens provided")
    
    sentence = tokens_to_sentence(request.tokens)
    return InterpretResponse(
        tokens=request.tokens,
        sentence=sentence
    )


@app.on_event("shutdown")
def shutdown():
    recognizer.close()
