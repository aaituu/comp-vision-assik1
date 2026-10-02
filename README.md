# Action Recognition from Photos

Authors: Ayana Markhabat and Balgyn Yermakhanbet  
Group: SE-2404

This computer vision project classifies a single image as `sitting`, `standing`,
or `waving`. The dataset was collected from Wikimedia Commons and does not use
Kaggle. It contains 360 images, with 120 images in each class.

The project compares two models:

- HOG features with Logistic Regression as the baseline;
- ImageNet-pretrained MobileNetV3-Small as the transfer-learning model.

The saved test accuracy is 38.9% for the baseline and 77.8% for
MobileNetV3-Small.

## Start here

### 1. Clone the repository

```bash
git clone https://github.com/aaituu/comp-vision-assik1.git
cd comp-vision-assik1
```

### 2. Create a Python environment

Python 3.12 is recommended.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Train and evaluate the models

The dataset is already included, so this is the main command needed to reproduce
the experiment:

```bash
python src/train_models.py --data-dir data --output-dir results --epochs 8 --seed 42
```

The command trains both models and writes the following files to `results/`:

- `metrics.json` - accuracy, precision, recall, F1-score, and confusion matrices;
- `test_predictions.csv` - prediction for every test image;
- `mobilenet_v3_small_best.pt` - best transfer-learning checkpoint;
- `confusion_baseline.png` and `confusion_transfer.png`;
- `training_history.png`, `class_distribution.png`, and `dataset_examples.png`;
- `data_splits.csv` - fixed train, validation, and test split.

The first training run downloads the official MobileNetV3-Small ImageNet
weights. Internet access is required only for that download.

### 4. Open the completed notebook

```bash
jupyter notebook notebooks/action_recognition.ipynb
```

The notebook is already executed and contains dataset analysis, preprocessing,
model descriptions, metrics, confusion matrices, error analysis, and the full
source code.

### 5. Build the report again

```bash
python scripts/build_report.py
```

The generated APA-style article is saved at:

```text
output/pdf/action_recognition_report.pdf
```

The report contains no page header, footer, or page number.

## Project structure

```text
comp-vision-assik1/
├── data/
│   ├── raw/
│   │   ├── sitting/              # 120 images
│   │   ├── standing/             # 120 images
│   │   └── waving/               # 120 images
│   ├── manifest.csv              # sources, authors, licenses, checksums
│   └── dataset_summary.json      # dataset counts and collection settings
├── notebooks/
│   └── action_recognition.ipynb  # completed experiment notebook
├── src/
│   └── train_models.py           # training and evaluation pipeline
├── scripts/
│   ├── collect_dataset.py        # Wikimedia Commons data collector
│   ├── build_notebook.py         # notebook generator
│   └── build_report.py           # APA-style PDF report generator
├── results/                      # metrics, plots, predictions, saved model
├── output/pdf/
│   └── action_recognition_report.pdf
├── DATASET_CARD.md               # dataset description and limitations
├── requirements.txt              # Python dependencies
├── dataset_action_recognition.zip
└── submission_action_recognition.zip
```

## Optional: collect the dataset again

The repository already contains the full dataset. Run the collector only when a
new download is required:

```bash
python scripts/collect_dataset.py --output data --per-class 120 --seed 42
```

This command needs internet access and `curl`. It updates the images,
`data/manifest.csv`, and `data/dataset_summary.json`.

## Rebuild and execute the notebook

```bash
python scripts/build_notebook.py
jupyter nbconvert --to notebook --execute notebooks/action_recognition.ipynb --inplace
```

## MobileNet weight download troubleshooting

If Python reports an SSL certificate error while downloading the pretrained
weights on macOS, download the same official file with `curl`:

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
curl -L https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth \
  -o ~/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
```

Then run the training command again.

## Data and licensing

The images come from Wikimedia Commons and have different free licenses or
public-domain status. `data/manifest.csv` contains the source page, image URL,
author or credit, license, original dimensions, and SHA-256 checksum for every
image. The original Wikimedia Commons page is the authoritative license record.
