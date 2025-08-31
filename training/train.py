import argparse, os, math, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.transforms import InterpolationMode
from collections import Counter
from tqdm import tqdm

EMOTIONS = ["angry","disgust","fear","happy","neutral","sad","surprise"]

def build_dataloaders(data_root, img_size, batch_size, num_workers=4):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    train_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(int(img_size*1.15), interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tfms)
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=val_tfms)
    test_ds = datasets.ImageFolder(os.path.join(data_root, "test"), transform=val_tfms)
    # Weighted sampler to mitigate class imbalance
    counts = Counter([y for _, y in train_ds.samples])
    cls_count = torch.tensor([counts[i] for i in range(len(train_ds.classes))], dtype=torch.float)
    cls_weights = 1.0 / cls_count
    sample_weights = torch.tensor([cls_weights[y] for _, y in train_ds.samples])
    sampler = WeightedRandomSampler(weights=sample_weights.double(), num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.classes

def build_model(model_name, num_classes, pretrained=True):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError("Unsupported model: " + model_name)
    return model

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for images, targets in tqdm(loader, leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, targets)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs, targets) * images.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss, running_acc = 0.0, 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs, targets) * images.size(0)
    n = len(loader.dataset)
    return running_loss / n, running_acc / n

def save_checkpoint(state, out_dir, is_best):
    os.makedirs(out_dir, exist_ok=True)
    fname = "best.pt" if is_best else f"epoch{state['epoch']:03d}.pt"
    torch.save(state, os.path.join(out_dir, fname))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="models/checkpoints/fer2013-resnet18")
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, classes = build_dataloaders(args.data_root, args.img_size, args.batch_size, args.num_workers)
    model = build_model(args.model, num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and device.type == "cuda") else None

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        print(f"  train: loss={tr_loss:.4f} acc={tr_acc:.4f}")
        print(f"  valid: loss={val_loss:.4f} acc={val_acc:.4f}")
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "classes": classes,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "model_name": args.model,
            "img_size": args.img_size,
        }, args.out_dir, is_best=is_best)

    # Final evaluation on test set using best checkpoint
    best_ckpt = os.path.join(args.out_dir, "best.pt")
    if os.path.exists(best_ckpt):
        print(f"Loading best checkpoint: {best_ckpt}")
        state = torch.load(best_ckpt, map_location=device)
        model = build_model(state.get("model_name", args.model), num_classes=len(classes))
        model.load_state_dict(state["state_dict"])
        model.to(device)
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"TEST: loss={test_loss:.4f} acc={test_acc:.4f}")

if __name__ == "__main__":
    main()