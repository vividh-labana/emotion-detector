import argparse, os, csv, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from train import build_model

EMOTIONS = ["angry","disgust","fear","happy","neutral","sad","surprise"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input-folder", dest="input_folder", required=True)
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    classes = state.get("classes", EMOTIONS)
    model = build_model(state.get("model_name","resnet18"), num_classes=len(classes))
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()

    tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(int(args.img_size*1.15), interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    images = []
    for ext in ("*.png","*.jpg","*.jpeg"):
        images.extend(Path(args.input_folder).glob(ext))

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image","pred_label","probability"])
        softmax = torch.nn.Softmax(dim=1)
        for img_path in images:
            img = Image.open(img_path).convert("L")
            x = tfms(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = softmax(logits)
                prob, pred = probs.max(dim=1)
            writer.writerow([str(img_path), classes[int(pred)], float(prob)])

    print(f"Saved predictions to {args.output}")

if __name__ == "__main__":
    main()