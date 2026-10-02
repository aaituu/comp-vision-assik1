#!/usr/bin/env python3
"""Create the final five-page project report as a plain A4 PDF."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "output" / "pdf" / "action_recognition_report.pdf"


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "Heading1Plain",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.black,
            spaceBefore=8,
            spaceAfter=5,
        ),
        "h2": ParagraphStyle(
            "Heading2Plain",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.black,
            spaceBefore=6,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "BodyPlain",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.1,
            alignment=TA_JUSTIFY,
            textColor=colors.black,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallPlain",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.black,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "CaptionPlain",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceBefore=2,
            spaceAfter=6,
        ),
    }


def paragraph(text: str, style):
    return Paragraph(text, style)


def simple_table(data, widths, font_size=8.5):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def add_page_number(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#888888"))
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.black)
    canvas.drawString(20 * mm, 10 * mm, "Action Recognition from Photos")
    canvas.drawRightString(190 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def build() -> Path:
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    baseline = metrics["baseline"]
    transfer = metrics["transfer_learning"]
    s = styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="Action Recognition from Photos",
        author="Computer Vision Course Project",
    )
    story = []

    # Page 1
    story.append(paragraph("Action Recognition from Photos", s["title"]))
    story.append(paragraph("Sitting, Standing, and Waving", s["subtitle"]))
    story.append(paragraph("Abstract", s["h1"]))
    story.append(
        paragraph(
            "This project studies action recognition from one photo. The task is to classify "
            "an image as sitting, standing, or waving. I collected a new dataset from "
            "Wikimedia Commons instead of using Kaggle or another ready-made dataset. The "
            "final dataset has 360 images, with 120 images in each class. I compared a simple "
            "baseline based on HOG features and Logistic Regression with a transfer-learning "
            "model based on MobileNetV3-Small. On the fixed test set, the baseline reached "
            f"{baseline['accuracy'] * 100:.1f}% accuracy and the transfer model reached "
            f"{transfer['accuracy'] * 100:.1f}%. The result shows that pretrained CNN features "
            "are much more useful for this varied image dataset.",
            s["body"],
        )
    )
    story.append(paragraph("1. Introduction", s["h1"]))
    story.append(
        paragraph(
            "Recognizing human actions is useful in photo organization, accessibility tools, "
            "human-computer interaction, and scene understanding. It is harder than simple "
            "object recognition because the important information is often the position of the "
            "whole body or a small part such as a raised hand. A single photo also does not show "
            "motion over time, so context and body pose become important.",
            s["body"],
        )
    )
    story.append(
        paragraph(
            "The goal of this assignment was to complete the full computer vision process: "
            "collect data, clean and organize it, apply preprocessing and augmentation, train "
            "two models, and evaluate them with accuracy, precision, recall, F1-score, and "
            "confusion matrices. The same train, validation, and test split was used for a fair "
            "comparison.",
            s["body"],
        )
    )
    story.append(paragraph("Project summary", s["h2"]))
    story.append(
        simple_table(
            [
                ["Item", "Value"],
                ["Classes", "sitting, standing, waving"],
                ["Images", "360 total, 120 per class"],
                ["Split", "252 train, 54 validation, 54 test"],
                ["Baseline", "HOG + Logistic Regression"],
                ["Improved model", "MobileNetV3-Small transfer learning"],
            ],
            [55 * mm, 100 * mm],
        )
    )
    story.append(PageBreak())

    # Page 2
    story.append(paragraph("2. Dataset Collection", s["h1"]))
    story.append(
        paragraph(
            "I collected the images from Wikimedia Commons with the official MediaWiki API. "
            "The source categories were People sitting, People standing, Female people waving "
            "hands, and Male people waving hands. This means the dataset was collected for this "
            "project and was not downloaded as a prepared machine-learning dataset. A fixed seed "
            "was used when selecting candidates, which makes the collection repeatable.",
            s["body"],
        )
    )
    story.append(
        paragraph(
            "The script accepted JPEG, PNG, and WebP files, converted them to RGB JPEG, and "
            "limited the longest image side to 640 pixels. Images smaller than 180 pixels on "
            "either side were rejected. Exact duplicates were removed by SHA-256 and close "
            "duplicates were checked with a perceptual difference hash. The manifest records "
            "the original page, image URL, author or credit, license, dimensions, and checksum "
            "for every file. All 360 records have a source page and a license value.",
            s["body"],
        )
    )
    story.append(Image(str(RESULTS / "dataset_examples.png"), width=150 * mm, height=112.5 * mm))
    story.append(paragraph("Figure 1. Example images from the three classes.", s["caption"]))
    story.append(Image(str(RESULTS / "class_distribution.png"), width=90 * mm, height=54 * mm))
    story.append(paragraph("Figure 2. Balanced class distribution.", s["caption"]))
    story.append(PageBreak())

    # Page 3
    story.append(paragraph("3. Preprocessing and Models", s["h1"]))
    story.append(paragraph("3.1 Data split and augmentation", s["h2"]))
    story.append(
        paragraph(
            "I used a stratified 70/15/15 split with random seed 42. Each split contains the "
            "same number of examples from every class: 84 per class for training, 18 for "
            "validation, and 18 for testing. The test images were kept separate until the final "
            "evaluation.",
            s["body"],
        )
    )
    story.append(
        paragraph(
            "For the baseline, images were resized to 128 x 128 and converted to grayscale. "
            "For the CNN, training images used random resized crop, horizontal flip, rotation "
            "up to 8 degrees, and small brightness and contrast changes. These operations create "
            "slightly different examples during training and reduce overfitting. Validation and "
            "test images used resize to 256, center crop to 224 x 224, conversion to tensor, and "
            "ImageNet normalization.",
            s["body"],
        )
    )
    story.append(paragraph("3.2 Baseline model", s["h2"]))
    story.append(
        paragraph(
            "The baseline uses Histograms of Oriented Gradients (HOG). HOG describes local edge "
            "directions and is useful for human shape. The HOG vectors were standardized and "
            "given to Logistic Regression. This model is fast and gives a clear reference point, "
            "but it cannot learn high-level visual features from the data.",
            s["body"],
        )
    )
    story.append(paragraph("3.3 Transfer-learning model", s["h2"]))
    story.append(
        paragraph(
            "The improved model is MobileNetV3-Small with ImageNet pretrained weights. I froze "
            "the convolutional feature extractor and replaced the last layer with a three-class "
            "linear layer. The classifier was trained for 8 epochs with AdamW, learning rate "
            "0.001, weight decay 0.0001, batch size 16, and cross-entropy loss. The checkpoint "
            "with the highest validation accuracy was used for testing. Training ran on Apple "
            "MPS acceleration.",
            s["body"],
        )
    )
    story.append(Image(str(RESULTS / "training_history.png"), width=112 * mm, height=70 * mm))
    story.append(
        paragraph(
            f"Figure 3. Training history. Best validation accuracy was "
            f"{transfer['best_validation_accuracy'] * 100:.1f}%.",
            s["caption"],
        )
    )
    story.append(PageBreak())

    # Page 4
    story.append(paragraph("4. Evaluation Results", s["h1"]))
    story.append(
        paragraph(
            "The improved model performed much better on every main metric. Its accuracy was "
            f"{transfer['accuracy'] * 100:.1f}%, compared with {baseline['accuracy'] * 100:.1f}% "
            "for the baseline. Macro F1 is important here because it gives equal weight to all "
            "three classes. The transfer model improved macro F1 from "
            f"{baseline['f1_macro']:.3f} to {transfer['f1_macro']:.3f}.",
            s["body"],
        )
    )
    story.append(
        simple_table(
            [
                ["Model", "Accuracy", "Precision", "Recall", "F1"],
                [
                    "HOG + Logistic Regression",
                    f"{baseline['accuracy']:.3f}",
                    f"{baseline['precision_macro']:.3f}",
                    f"{baseline['recall_macro']:.3f}",
                    f"{baseline['f1_macro']:.3f}",
                ],
                [
                    "MobileNetV3-Small",
                    f"{transfer['accuracy']:.3f}",
                    f"{transfer['precision_macro']:.3f}",
                    f"{transfer['recall_macro']:.3f}",
                    f"{transfer['f1_macro']:.3f}",
                ],
            ],
            [62 * mm, 24 * mm, 24 * mm, 24 * mm, 24 * mm],
        )
    )
    story.append(Spacer(1, 4 * mm))
    class_report = transfer["classification_report"]
    story.append(paragraph("MobileNetV3-Small results by class", s["h2"]))
    story.append(
        simple_table(
            [
                ["Class", "Precision", "Recall", "F1", "Test images"],
                *[
                    [
                        name,
                        f"{class_report[name]['precision']:.3f}",
                        f"{class_report[name]['recall']:.3f}",
                        f"{class_report[name]['f1-score']:.3f}",
                        str(int(class_report[name]["support"])),
                    ]
                    for name in metrics["class_names"]
                ],
            ],
            [38 * mm, 28 * mm, 28 * mm, 28 * mm, 30 * mm],
        )
    )
    story.append(Spacer(1, 5 * mm))
    matrix_table = Table(
        [
            [
                Image(str(RESULTS / "confusion_baseline.png"), width=77 * mm, height=65 * mm),
                Image(str(RESULTS / "confusion_transfer.png"), width=77 * mm, height=65 * mm),
            ]
        ],
        colWidths=[80 * mm, 80 * mm],
    )
    matrix_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(matrix_table)
    story.append(paragraph("Figure 4. Confusion matrices for both models.", s["caption"]))
    story.append(PageBreak())

    # Page 5
    story.append(paragraph("5. Discussion", s["h1"]))
    story.append(
        paragraph(
            "The baseline was only slightly above the 33.3% random level. Its confusion matrix "
            "shows that it mixed all three classes. HOG mainly describes edges, so it can react "
            "to background structure, clothing, or image style instead of understanding the full "
            "pose. The transfer model correctly classified 42 of 54 test images. It performed "
            "best on sitting by precision and on waving by recall. Standing was the hardest class, "
            "with three examples predicted as waving and two as sitting.",
            s["body"],
        )
    )
    story.append(
        paragraph(
            "The validation accuracy changed between epochs while training loss continued to go "
            "down. This suggests some overfitting, which is expected with only 252 training images. "
            "Saving the best validation checkpoint helped avoid using the last epoch automatically. "
            "The large improvement on the test set still shows the value of transfer learning.",
            s["body"],
        )
    )
    story.append(paragraph("5.1 Limitations and possible improvements", s["h2"]))
    story.append(
        paragraph(
            "The dataset was collected from public internet categories, so the labels were not "
            "checked by several independent annotators. Some images contain several people, a "
            "small person, or more than one possible action. The collection also includes different "
            "periods, cameras, and image styles. These factors make the task realistic but add label "
            "noise. The test set has only 18 images per class, so a few mistakes change the score.",
            s["body"],
        )
    )
    story.append(
        paragraph(
            "A stronger next version would manually review all labels, crop the main person, collect "
            "more modern photos, and group related images before splitting to reduce source similarity. "
            "It would also be useful to fine-tune the last MobileNet feature blocks with a smaller "
            "learning rate and use cross-validation. For waving, short video clips could provide motion "
            "information that is missing from one photo.",
            s["body"],
        )
    )
    story.append(paragraph("6. Conclusion", s["h1"]))
    story.append(
        paragraph(
            "This project completed the required data collection, preprocessing, augmentation, model "
            "comparison, and evaluation steps. A balanced 360-image dataset was built without Kaggle. "
            f"MobileNetV3-Small reached {transfer['accuracy'] * 100:.1f}% test accuracy and clearly "
            "outperformed the HOG baseline. The main lesson is that pretrained CNN features can learn "
            "useful pose information even when the custom dataset is small and visually varied.",
            s["body"],
        )
    )
    story.append(paragraph("References", s["h2"]))
    references = [
        "[1] Wikimedia Commons, MediaWiki API and Imageinfo documentation, https://www.mediawiki.org/wiki/API:Imageinfo",
        "[2] N. Dalal and B. Triggs, Histograms of Oriented Gradients for Human Detection, CVPR, 2005, https://doi.org/10.1109/CVPR.2005.177",
        "[3] A. Howard et al., Searching for MobileNetV3, ICCV, 2019, https://arxiv.org/abs/1905.02244",
        "[4] PyTorch, torchvision MobileNetV3-Small documentation, https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html",
    ]
    for item in references:
        story.append(paragraph(item, s["small"]))

    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return OUTPUT


if __name__ == "__main__":
    print(build())
