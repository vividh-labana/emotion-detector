# Training Pipeline (FER2013)

This folder adds a **from-scratch training pipeline** for facial emotion recognition using the **FER2013** dataset.

## Dataset (FER2013)
FER2013 contains 48×48 grayscale face images labeled with seven emotions: *angry, disgust, fear, happy, sad, surprise, neutral*.
You can obtain it from Kaggle and place the files in `training/datasets/FER2013/raw/` (e.g., `fer2013.csv` or the `train/` and `test/` folders).

### Option A — Kaggle API (preferred)
```bash
pip install kaggle
# Put kaggle.json in ~/.kaggle/ with your Kaggle API token
kaggle datasets download -d msambare/fer2013 -p training/datasets/FER2013/raw --unzip
```
This may give you either a CSV or already-separated image folders. The prep script handles both.

### Prepare the foldered dataset
```bash
# From repo root
python training/prepare_fer2013.py   --raw-dir training/datasets/FER2013/raw   --out-dir training/datasets/FER2013/images   --val-size 0.1 --seed 42
```
After this, you should have:
```
training/datasets/FER2013/images/
├─ train/{angry,disgust,fear,happy,neutral,sad,surprise}/
├─ val/{...}/
└─ test/{...}/
```

## Train
```bash
pip install -r training/requirements-train.txt
python training/train.py --data-root training/datasets/FER2013/images   --epochs 25 --batch-size 128 --lr 1e-3 --model resnet18 --img-size 224   --mixed-precision --out-dir models/checkpoints/fer2013-resnet18
```

## Evaluate
```bash
python training/eval.py --data-root training/datasets/FER2013/images   --checkpoint models/checkpoints/fer2013-resnet18/best.pt
```

## Inference on a folder
```bash
python training/infer.py --checkpoint models/checkpoints/fer2013-resnet18/best.pt   --input-folder path/to/images --output predictions.csv
```

## Notes
- Models are **for demo/education**. Emotion is subjective; results can be biased/wrong. Do not use for decisions about people.
- For better performance, consider "FER+" labels or other datasets later. See docs in the repo root.