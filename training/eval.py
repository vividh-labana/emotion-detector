import argparse, os, torch
from pathlib import Path
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from train import build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(int(args.img_size*1.15), interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    test_ds = datasets.ImageFolder(os.path.join(args.data_root, "test"), transform=tfms)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    state = torch.load(args.checkpoint, map_location=device)
    model = build_model(state.get("model_name","resnet18"), num_classes=len(test_ds.classes))
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == targets).sum().item()
            total += images.size(0)
    print(f"Test loss: {loss_sum/total:.4f}, acc: {correct/total:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()