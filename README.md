# 🎭 Emotion Detector with Custom Training

A comprehensive emotion detection application that combines **text sentiment analysis** and **facial emotion recognition** using both pre-trained models and custom-trained PyTorch models.

## 🌟 Features

- **📝 Text Emotion Analysis**: Analyze emotions in text using Hugging Face transformers
- **😊 Face Emotion Detection**: Detect emotions from facial expressions using a custom-trained ResNet18 model
- **🎯 Custom Model Training**: Train your own emotion detection models on the FER2013 dataset
- **🖥️ Interactive Web Interface**: User-friendly Streamlit application
- **⚡ High Performance**: Custom-trained model with 87.5% accuracy on happy emotion detection

## 🚀 Quick Start

### Prerequisites

- **Python 3.8 or higher** (Python 3.12 recommended)
- **Git** (for cloning the repository)
- **Internet connection** (for downloading dependencies)

### 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vividh-labana/emotion-detector.git
   cd emotion-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 🎯 Usage

### Web Interface

The Streamlit app provides two main features:

1. **Text Emotions Tab**
   - Enter any text to analyze its emotional content
   - Get detailed emotion scores and confidence levels
   - Powered by Hugging Face transformers

2. **Face Emotions Tab**
   - Upload an image containing faces
   - Get emotion predictions for each detected face
   - Powered by custom-trained ResNet18 model

### Supported Emotions

The face emotion detector recognizes **7 emotions**:
- 😠 Angry
- 🤢 Disgust  
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

## 🧠 Model Information

### Custom-Trained Face Emotion Model

- **Architecture**: ResNet18
- **Dataset**: FER2013 (35,887 images)
- **Training**: 25 epochs with mixed precision
- **Performance**: 87.5% accuracy on happy emotion detection
- **Model Size**: ~43MB

### Text Emotion Model

- **Model**: Hugging Face transformers
- **Capabilities**: Multi-emotion classification
- **Languages**: Primarily English

## 🔧 Advanced Usage

### Training Your Own Model

If you want to retrain the emotion detection model:

1. **Install training dependencies**
   ```bash
   cd training
   pip install -r requirements-train.txt
   ```

2. **Set up Kaggle API** (for dataset download)
   - Get your `kaggle.json` from [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Place it in `~/.kaggle/kaggle.json` (create the folder if needed)
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download and prepare dataset**
   ```bash
   kaggle datasets download -d msambare/fer2013 -p datasets/FER2013/raw --unzip
   python prepare_fer2013.py --raw-dir datasets/FER2013/raw --out-dir datasets/FER2013/images --val-size 0.10 --seed 42
   ```

4. **Train the model**
   ```bash
   python train.py --data-root datasets/FER2013/images --epochs 25 --batch-size 128 --lr 1e-3 --model resnet18 --img-size 224 --mixed-precision --out-dir models/checkpoints/fer2013-resnet18
   ```

5. **Evaluate the model**
   ```bash
   python eval.py --data-root datasets/FER2013/images --checkpoint models/checkpoints/fer2013-resnet18/best.pt
   ```

### Using the Model Programmatically

```python
from detectors.face_emotion import detect
from detectors.text_emotion import detect as detect_text
import numpy as np
from PIL import Image

# Face emotion detection
image = np.array(Image.open("path/to/image.jpg"))
face_results = detect(image)

# Text emotion detection  
text_results = detect_text("I am feeling great today!")
```

## 📁 Project Structure

```
emotion-detector/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Main dependencies
├── README.md                       # This file
├── detectors/                      # Detection modules
│   ├── __init__.py
│   ├── face_emotion.py            # Face emotion detection
│   ├── face_model_infer.py        # PyTorch model loader
│   └── text_emotion.py            # Text emotion detection
├── models/                         # Trained models
│   └── checkpoints/
│       └── fer2013-resnet18/
│           └── best.pt            # Best trained model (43MB)
└── training/                       # Training scripts and data
    ├── train.py                   # Model training script
    ├── eval.py                    # Model evaluation script
    ├── infer.py                   # Inference script
    ├── prepare_fer2013.py         # Dataset preparation
    ├── requirements-train.txt     # Training dependencies
    └── datasets/                  # Dataset storage
```

## 🛠️ Technical Details

### Dependencies

**Main Application:**
- `streamlit>=1.36.0` - Web interface
- `torch>=2.0.0` - PyTorch for model inference
- `torchvision>=0.15.0` - Image preprocessing
- `transformers>=4.40.0` - Text emotion analysis
- `opencv-python-headless>=4.8.0.74` - Face detection
- `pillow>=10.2.0` - Image processing
- `numpy>=1.24.0` - Numerical computations
- `pandas>=2.0.0` - Data handling

**Training (Optional):**
- `scikit-learn>=1.3.0` - Metrics and evaluation
- `tqdm>=4.66.0` - Progress bars
- `kaggle` - Dataset download
- `PyYAML>=6.0.0` - Configuration files

### System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **CPU**: Any modern processor (GPU optional for training)
- **OS**: Windows 10/11, macOS, or Linux

## 🚨 Troubleshooting

### Common Issues

1. **"streamlit: command not found"**
   ```bash
   pip install streamlit
   ```

2. **"No module named 'torch'"**
   ```bash
   pip install torch torchvision
   ```

3. **Model loading errors**
   - Ensure `training/models/checkpoints/fer2013-resnet18/best.pt` exists
   - Check file permissions and path

4. **Memory issues during training**
   - Reduce batch size: `--batch-size 64` or `--batch-size 32`
   - Use CPU only: Remove `--mixed-precision` flag

### Performance Tips

- **For faster inference**: Use GPU if available
- **For lower memory usage**: Close other applications
- **For better accuracy**: Use higher resolution images (224x224 minimum)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m "Add feature"`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **FER2013 Dataset**: [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **Hugging Face**: Transformers library
- **OpenCV**: Computer vision library

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/vividh-labana/emotion-detector/issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Made with ❤️ by [vividh-labana](https://github.com/vividh-labana)**