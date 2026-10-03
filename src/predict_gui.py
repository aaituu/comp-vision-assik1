from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageOps, ImageTk
from torch import nn
from torchvision import models, transforms


CLASSES = ["sitting", "standing", "waving"]
CLASS_NAMES = {
    "sitting": "Sitting",
    "standing": "Standing",
    "waving": "Waving",
}
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "results" / "mobilenet_v3_small_best.pt"


def evaluation_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(model_path: Path, device: torch.device):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_image(model, image: Image.Image, device: torch.device) -> list[float]:
    tensor = evaluation_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    return probabilities.cpu().tolist()


class PredictionApp:
    def __init__(self, root: tk.Tk, model_path: Path):
        self.root = root
        self.root.title("Action Recognition from a Photo")
        self.root.geometry("720x680")
        self.root.minsize(620, 620)
        self.root.configure(background="#f3f4f6")

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.model = load_model(model_path, self.device)
        self.preview_photo = None

        self._build_interface()

    def _build_interface(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Arial", 22, "bold"))
        style.configure("Result.TLabel", font=("Arial", 20, "bold"))
        style.configure("Info.TLabel", font=("Arial", 12))
        style.configure("Select.TButton", font=("Arial", 13, "bold"), padding=10)

        container = ttk.Frame(self.root, padding=24)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Action Recognition",
            style="Title.TLabel",
        ).pack(pady=(0, 6))
        ttk.Label(
            container,
            text="Choose a photo of a person sitting, standing, or waving",
            style="Info.TLabel",
        ).pack(pady=(0, 18))

        self.image_label = ttk.Label(
            container,
            text="No photo selected",
            anchor="center",
            relief="solid",
            padding=10,
        )
        self.image_label.pack(fill="both", expand=True, pady=(0, 18))

        ttk.Button(
            container,
            text="Choose Photo",
            command=self.choose_image,
            style="Select.TButton",
        ).pack()

        self.result_label = ttk.Label(
            container,
            text="",
            style="Result.TLabel",
            anchor="center",
        )
        self.result_label.pack(pady=(18, 8))

        self.details_label = ttk.Label(
            container,
            text="",
            style="Info.TLabel",
            justify="left",
        )
        self.details_label.pack()

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a Photo",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            probabilities = predict_image(self.model, image, self.device)
            self._show_image(image)
            self._show_result(probabilities)
        except Exception as error:
            messagebox.showerror(
                "Unable to Analyze the Photo",
                f"Make sure you selected a valid image file.\n\n{error}",
            )

    def _show_image(self, image: Image.Image) -> None:
        preview = image.copy()
        preview.thumbnail((560, 360), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self.preview_photo, text="")

    def _show_result(self, probabilities: list[float]) -> None:
        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        best_class = CLASSES[best_index]
        best_percent = probabilities[best_index] * 100
        self.result_label.configure(
            text=f"Prediction: {CLASS_NAMES[best_class]} — {best_percent:.1f}%"
        )

        lines = [
            f"{CLASS_NAMES[class_name]}: {probability * 100:.1f}%"
            for class_name, probability in zip(CLASSES, probabilities)
        ]
        self.details_label.configure(text="\n".join(lines))


def main() -> None:
    model_path = DEFAULT_MODEL_PATH
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1]).expanduser().resolve()

    if not model_path.is_file():
        print(f"Model file was not found: {model_path}", file=sys.stderr)
        print("Run src/train_models.py first.", file=sys.stderr)
        raise SystemExit(1)

    root = tk.Tk()
    try:
        PredictionApp(root, model_path)
    except Exception as error:
        root.withdraw()
        messagebox.showerror("Model Loading Error", str(error))
        root.destroy()
        raise SystemExit(1) from error
    root.mainloop()


if __name__ == "__main__":
    main()
