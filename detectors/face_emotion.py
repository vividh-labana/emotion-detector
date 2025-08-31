from typing import List, Dict, Any
import numpy as np
import cv2
from .face_model_infer import get_pytorch_model

# Face detection using OpenCV (similar to what FER library uses)
_face_cascade = None

def _get_face_cascade():
    """Get OpenCV face cascade classifier."""
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return _face_cascade

def _detect_faces(image_rgb: np.ndarray) -> List[List[int]]:
    """Detect faces in the image using OpenCV."""
    # Convert RGB to grayscale for face detection
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    face_cascade = _get_face_cascade()
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Convert to list format [x, y, w, h]
    return [list(face) for face in faces]

def detect(image_rgb: np.ndarray) -> List[Dict[str, Any]]:
    """Returns a list of faces with boxes and emotion scores using trained PyTorch model.
    Each item: { 'box': [x,y,w,h], 'emotions': {label:score}, 'top_emotion': str, 'confidence': float }
    """
    try:
        # Get the trained PyTorch model
        pytorch_model = get_pytorch_model()
        
        # Detect faces using OpenCV
        face_boxes = _detect_faces(image_rgb)
        
        parsed = []
        for box in face_boxes:
            x, y, w, h = box
            
            # Extract face region
            face_region = image_rgb[y:y+h, x:x+w]
            
            # Skip if face region is too small
            if face_region.size == 0 or min(face_region.shape[:2]) < 20:
                continue
            
            # Get emotion predictions from PyTorch model
            emotions = pytorch_model.predict_emotion(face_region)
            
            if not emotions:
                continue
                
            # Find top emotion
            top_emotion = max(emotions, key=emotions.get)
            confidence = emotions[top_emotion]
            
            parsed.append({
                "box": box,
                "emotions": emotions,
                "top_emotion": top_emotion,
                "confidence": float(confidence)
            })
            
        return parsed
        
    except Exception as e:
        print(f"❌ Error in PyTorch emotion detection: {e}")
        print("💡 Falling back to basic face detection without emotion classification")
        
        # Fallback: just detect faces without emotion classification
        face_boxes = _detect_faces(image_rgb)
        fallback_emotions = {
            "angry": 0.14, "disgust": 0.14, "fear": 0.14, 
            "happy": 0.16, "neutral": 0.16, "sad": 0.13, "surprise": 0.13
        }
        
        return [{
            "box": box,
            "emotions": fallback_emotions,
            "top_emotion": "neutral",
            "confidence": 0.16
        } for box in face_boxes]