import argparse, os, csv, io, math, shutil, random
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

EMOTIONS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

def save_image(arr: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)

def parse_csv(csv_path: Path, out_dir: Path):
    # FER2013 CSV columns: emotion, pixels, Usage (Training/PublicTest/PrivateTest)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            emo_id = int(row["emotion"])
            usage = row.get("Usage", "Training")
            label = EMOTIONS.get(emo_id, "unknown")
            pixels = np.array(list(map(int, row["pixels"].split())), dtype=np.uint8).reshape(48, 48)
            img = pixels  # grayscale
            # Save into temp split dirs per usage first; we'll re-split val later
            split = "train" if usage == "Training" else ("test" if "Test" in usage else "train")
            out_path = out_dir / split / label / f"{i:07d}.png"
            save_image(img, out_path)

def copy_from_folders(raw_dir: Path, out_dir: Path):
    # If raw_dir already contains train/ test with class folders, copy and create val split
    for split in ["train", "test"]:
        src_split = raw_dir / split
        if not src_split.exists():
            continue
        for cls in sorted([p.name for p in src_split.iterdir() if p.is_dir()]):
            dst = out_dir / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for imgp in (src_split / cls).glob("*.*"):
                shutil.copy2(imgp, dst / imgp.name)

def ensure_val_split(out_dir: Path, val_size: float, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # For each class, move a fraction of train images to val
    for cls_dir in (out_dir / "train").glob("*"):
        if not cls_dir.is_dir():
            continue
        imgs = sorted([p for p in cls_dir.glob("*.png")] + [p for p in cls_dir.glob("*.jpg")] + [p for p in cls_dir.glob("*.jpeg")])
        if not imgs:
            continue
        n_val = max(1, int(len(imgs) * val_size))
        val_imgs = set(random.sample(imgs, n_val))
        (out_dir / "val" / cls_dir.name).mkdir(parents=True, exist_ok=True)
        for p in val_imgs:
            shutil.move(str(p), str(out_dir / "val" / cls_dir.name / p.name))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="Directory with raw FER2013 (csv or train/test folders)")
    ap.add_argument("--out-dir", required=True, help="Output directory for foldered images")
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        print(f"[WARN] {out_dir} exists; files may be overwritten/merged.")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "fer2013.csv"
    has_csf = csv_path.exists()

    if csv_path.exists():
        print(f"[INFO] Preparing from CSV: {csv_path}")
        parse_csv(csv_path, out_dir)
    else:
        print(f"[INFO] No CSV found at {csv_path}. Trying to copy from train/test folders...")
        copy_from_folders(raw_dir, out_dir)

    # Ensure val split exists
    ensure_val_split(out_dir, args.val_size, args.seed)

    print(f"[DONE] Prepared dataset under: {out_dir}")

if __name__ == "__main__":
    main()