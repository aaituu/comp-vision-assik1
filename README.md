# Action Recognition from Photos

Authors: Ayana Markhabat and Balgyn Yermakhanbet  
Group: SE-2404

The project classifies one photo as `sitting`, `standing`, or `waving`.
It contains 360 Wikimedia Commons images: 120 images in each class.

Two models are trained and compared:

- HOG features with Logistic Regression (baseline);
- ImageNet-pretrained MobileNetV3-Small (main model).

Saved test accuracy: 38.9% for the baseline and 77.8% for MobileNetV3-Small.

## Installation

Python 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```bat
.venv\Scripts\activate
```

## Train the models

```bash
python src/train_models.py --data-dir data --output-dir results --epochs 8 --seed 42
```

The training script saves the best MobileNet model, metrics, predictions, data
splits, and plots in `results/`.

The first training run may need internet access to download the official
ImageNet weights. The already trained model is included in the repository.

## Check a new photo

Activate the environment and open the graphical application:

```bash
source .venv/bin/activate
python src/predict_gui.py
```

Press **Choose photo**, select a JPG, PNG, BMP, or WebP file, and the application
will show the predicted action and the model probability for all three classes.

The application uses `results/mobilenet_v3_small_best.pt`, so retraining is not
required before every launch.

## Project structure

```text
├── data/                         # dataset and manifest
├── results/                      # trained model, metrics, predictions, plots
├── src/
│   ├── train_models.py           # model training and evaluation
│   └── predict_gui.py            # graphical prediction application
├── README.md
└── requirements.txt
```

The model knows only three classes. If an unrelated image or a different action
is selected, it will still choose one of `sitting`, `standing`, or `waving`.
