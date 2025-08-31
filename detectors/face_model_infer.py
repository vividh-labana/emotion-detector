import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from typing import Dict, Optional
import cv2

class PyTorchEmotionModel:
    """Helper class to load and use the trained PyTorch emotion detection model."""
    
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        self.transforms = None
        self._load_model(checkpoint_path)
    
    def _build_model(self, model_name: str, num_classes: int) -> torch.nn.Module:
        """Build the model architecture."""
        if model_name == "resnet18":
            model = models.resnet18(weights=None)  # No pretrained weights needed for inference
            in_feats = model.fc.in_features
            model.fc = torch.nn.Linear(in_feats, num_classes)
        elif model_name == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=None)
            in_feats = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feats, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return model
    
    def _load_model(self, checkpoint_path: str):
        """Load the trained model from checkpoint."""
        try:
            state = torch.load(checkpoint_path, map_location=self.device)
            self.classes = state.get("classes", self.classes)
            model_name = state.get("model_name", "resnet18")
            img_size = state.get("img_size", 224)
            
            # Build and load model
            self.model = self._build_model(model_name, len(self.classes))
            self.model.load_state_dict(state["state_dict"])
            self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self.transforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(int(img_size * 1.15), interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
            
            print(f"✅ Loaded PyTorch emotion model: {model_name} with {len(self.classes)} classes")
            
        except Exception as e:
            print(f"❌ Error loading PyTorch model: {e}")
            raise
    
    def predict_emotion(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Predict emotion from a face image.
        
        Args:
            face_image: RGB numpy array of the face region
            
        Returns:
            Dictionary with emotion probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Convert numpy array to PIL Image
            if face_image.dtype != np.uint8:
                face_image = (face_image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(face_image).convert("L")  # Convert to grayscale
            
            # Apply transforms
            input_tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                probs = probabilities.cpu().numpy()[0]
            
            # Create emotion dictionary
            emotion_dict = {emotion: float(prob) for emotion, prob in zip(self.classes, probs)}
            
            return emotion_dict
            
        except Exception as e:
            print(f"❌ Error in emotion prediction: {e}")
            # Return neutral prediction as fallback
            return {emotion: 1.0/len(self.classes) for emotion in self.classes}


# Global model instance
_pytorch_model: Optional[PyTorchEmotionModel] = None

def get_pytorch_model(checkpoint_path: str = "training/models/checkpoints/fer2013-resnet18/best.pt") -> PyTorchEmotionModel:
    """Get or create the global PyTorch emotion model instance."""
    global _pytorch_model
    if _pytorch_model is None:
        _pytorch_model = PyTorchEmotionModel(checkpoint_path)
    return _pytorch_model 