# Action Recognition from Photos

This course project classifies a single photo as `sitting`, `standing`, or
`waving`. The dataset contains 120 images per class collected from Wikimedia
Commons. Kaggle is not used. Every image has source and license information in
`data/manifest.csv`.

## Project files

- `data/raw/` - downloaded images arranged by class
- `data/manifest.csv` - source URL, author, license, dimensions, and checksum
- `notebooks/action_recognition.ipynb` - complete experiment notebook
- `src/train_models.py` - reproducible training and evaluation pipeline
- `results/` - metrics, plots, predictions, and saved transfer model
- `output/pdf/action_recognition_report.pdf` - final five-page report
- `dataset_action_recognition.zip` - submission-ready dataset archive

## Quick start

Clone the repository and enter the project folder:

```bash
git clone https://github.com/aaituu/comp-vision-assik1.git
cd comp-vision-assik1
```

Python 3.12 is recommended. Create a virtual environment and install the
packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The dataset and saved results are already included. To reproduce training:

```bash
python src/train_models.py --data-dir data --output-dir results --epochs 8 --seed 42
```

The first run downloads the official ImageNet weights for MobileNetV3-Small.
New metrics, plots, predictions, and the best checkpoint are saved in `results/`.

Open the completed notebook with:

```bash
jupyter notebook notebooks/action_recognition.ipynb
```

## Optional: collect the dataset again

The repository already contains all 360 images. Run this only if the dataset
must be downloaded again from Wikimedia Commons:

```bash
python scripts/collect_dataset.py --output data --per-class 120 --seed 42
```

The collector requires internet access and the `curl` command. It updates the
images, `data/manifest.csv`, and `data/dataset_summary.json`.

## Models and outputs

The baseline uses HOG features and Logistic Regression. The improved model uses
ImageNet-pretrained MobileNetV3-Small with a new three-class output layer. Both
models use the same fixed test set. Saved metrics are in `results/metrics.json`,
and individual predictions are in `results/test_predictions.csv`.

## Data use note

Images come from Wikimedia Commons and have different free licenses or public
domain status. Attribution details are preserved in `data/manifest.csv`. Check
the individual source page before using an image outside this educational
project.
