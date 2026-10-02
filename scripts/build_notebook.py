#!/usr/bin/env python3
"""Build the submission notebook with saved experiment results."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def reference_cell(path: Path, heading: str):
    source = "\n".join(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("from __future__ import")
    )
    indented = "\n".join("    " + line for line in source.splitlines())
    return [
        nbf.v4.new_markdown_cell(
            f"### {heading}\n\nThe complete source is included below for submission. "
            "It is disabled in this display cell because the experiment was already run."
        ),
        nbf.v4.new_code_cell("if False:\n" + indented),
    ]


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    cells = [
        nbf.v4.new_markdown_cell(
            "# Action Recognition from Photos\n\n"
            "This notebook presents a three-class image classification project: sitting, "
            "standing, and waving. The dataset was collected from Wikimedia Commons and "
            "does not use Kaggle. The experiment compares a HOG + Logistic Regression "
            "baseline with MobileNetV3-Small transfer learning."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import csv, json\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'data').exists():\n"
            "    ROOT = ROOT.parent\n"
            "DATA_DIR = ROOT / 'data'\n"
            "RESULTS_DIR = ROOT / 'results'\n"
            "metrics = json.loads((RESULTS_DIR / 'metrics.json').read_text())\n"
            "print('Project root:', ROOT)"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Dataset\n\n"
            "I collected 120 images per class with a fixed script. Each local file has a "
            "source page, author or credit, license, dimensions, and checksum in the manifest."
        ),
        nbf.v4.new_code_cell(
            "manifest = pd.read_csv(DATA_DIR / 'manifest.csv')\n"
            "summary = manifest.groupby('class').size().rename('images').to_frame()\n"
            "display(summary)\n"
            "print('Total images:', len(manifest))\n"
            "print('Missing source pages:', manifest['source_page'].isna().sum())\n"
            "print('Missing licenses:', manifest['license'].isna().sum())"
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(RESULTS_DIR / 'dataset_examples.png'), width=800))\n"
            "display(Image(filename=str(RESULTS_DIR / 'class_distribution.png'), width=560))"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Preprocessing and split\n\n"
            "The data was divided with stratification into 70% training, 15% validation, "
            "and 15% test sets. The random seed was 42. For the baseline, images were resized "
            "to 128 x 128 and converted to grayscale before HOG extraction. For transfer "
            "learning, training images used random crop, horizontal flip, small rotation, and "
            "brightness/contrast change. Validation and test images used deterministic resize "
            "and center crop. All deep-model inputs used ImageNet normalization."
        ),
        nbf.v4.new_code_cell(
            "splits = pd.read_csv(RESULTS_DIR / 'data_splits.csv')\n"
            "display(pd.crosstab(splits['split'], splits['class']))\n"
            "print('Split sizes:', metrics['split_sizes'])"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Models\n\n"
            "The baseline is Logistic Regression trained on HOG features. This is a simple "
            "model that uses edge direction and shape information. The improved model is "
            "MobileNetV3-Small pretrained on ImageNet. Its feature extractor was frozen and "
            "the classifier was replaced with a three-class layer. The classifier was trained "
            "for eight epochs with AdamW and cross-entropy loss."
        ),
        nbf.v4.new_code_cell(
            "rows = []\n"
            "for key, name in [('baseline', 'HOG + Logistic Regression'), "
            "('transfer_learning', 'MobileNetV3-Small')]:\n"
            "    result = metrics[key]\n"
            "    rows.append({\n"
            "        'model': name,\n"
            "        'accuracy': result['accuracy'],\n"
            "        'macro precision': result['precision_macro'],\n"
            "        'macro recall': result['recall_macro'],\n"
            "        'macro F1': result['f1_macro'],\n"
            "    })\n"
            "comparison = pd.DataFrame(rows).set_index('model')\n"
            "display(comparison.round(3))"
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(RESULTS_DIR / 'training_history.png'), width=560))\n"
            "print('Best validation accuracy:', "
            "round(metrics['transfer_learning']['best_validation_accuracy'], 3))"
        ),
        nbf.v4.new_markdown_cell("## 4. Confusion matrices and class results"),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(RESULTS_DIR / 'confusion_baseline.png'), width=500))\n"
            "display(Image(filename=str(RESULTS_DIR / 'confusion_transfer.png'), width=500))"
        ),
        nbf.v4.new_code_cell(
            "class_rows = []\n"
            "report = metrics['transfer_learning']['classification_report']\n"
            "for class_name in metrics['class_names']:\n"
            "    class_rows.append({'class': class_name, **report[class_name]})\n"
            "display(pd.DataFrame(class_rows).set_index('class').round(3))"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. Error analysis\n\n"
            "The transfer model made fewer errors for every class. The main remaining "
            "confusion was between standing and the other classes. This is reasonable because "
            "some web photos contain several people or show a person only as a small part of "
            "the scene. Waving is also a short motion represented by one still image, so a "
            "raised hand can be ambiguous."
        ),
        nbf.v4.new_code_cell(
            "predictions = pd.read_csv(RESULTS_DIR / 'test_predictions.csv')\n"
            "predictions['baseline_correct'] = (predictions.true_class == predictions.baseline_prediction)\n"
            "predictions['transfer_correct'] = (predictions.true_class == predictions.transfer_prediction)\n"
            "print('Baseline errors:', (~predictions.baseline_correct).sum())\n"
            "print('Transfer model errors:', (~predictions.transfer_correct).sum())\n"
            "display(predictions.loc[~predictions.transfer_correct].head(12))"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Reproducing the experiment\n\n"
            "Set `RUN_TRAINING` to `True` to run the complete training script again. The "
            "default is false so opening the submitted notebook does not overwrite saved results."
        ),
        nbf.v4.new_code_cell(
            "RUN_TRAINING = False\n"
            "if RUN_TRAINING:\n"
            "    import subprocess, sys\n"
            "    subprocess.run([sys.executable, str(ROOT / 'src' / 'train_models.py'), "
            "'--data-dir', str(DATA_DIR), '--output-dir', str(RESULTS_DIR), "
            "'--epochs', '8', '--seed', '42'], check=True)"
        ),
        nbf.v4.new_markdown_cell(
            "## 7. Conclusion\n\n"
            "Transfer learning gave a clear improvement over the baseline. The final test "
            "accuracy was 77.8%, compared with 38.9% for HOG + Logistic Regression. This "
            "shows that pretrained visual features are useful for pose and action information "
            "in a small, varied dataset. More manual label checking and person cropping would "
            "probably improve the result further."
        ),
    ]
    cells.extend(reference_cell(ROOT / "scripts" / "collect_dataset.py", "Complete dataset collection code"))
    cells.extend(reference_cell(ROOT / "src" / "train_models.py", "Complete training and evaluation code"))
    notebook["cells"] = cells
    output = ROOT / "notebooks" / "action_recognition.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    return output


if __name__ == "__main__":
    print(build())
