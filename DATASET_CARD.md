# Dataset Card: Action Recognition from Photos

## Description

The dataset contains photographs for three classes: `sitting`, `standing`, and
`waving`. It was collected for a computer vision course assignment. There are
120 images in each class and 360 images in total.

## Source and collection

The images were downloaded from Wikimedia Commons through the MediaWiki API.
The source categories were `People sitting`, `People standing`, `Female people
waving hands`, and `Male people waving hands`. Kaggle was not used.

The collection script uses a fixed random seed, rejects files smaller than 180
pixels on either side, converts accepted files to RGB JPEG, limits the largest
side to 640 pixels, and removes exact and close perceptual duplicates inside
each class.

## Attribution

`data/manifest.csv` gives the original Wikimedia Commons file page, image URL,
author or credit, license name, license URL, original dimensions, and SHA-256
checksum for every local image. The images have different free licenses or
public-domain status. The individual source page remains the authoritative
license record.

## Intended use

The dataset is intended for this educational action-recognition experiment. It
can be used to reproduce the notebook and compare the HOG baseline with the
MobileNetV3-Small transfer-learning model.

## Limitations

The labels come from Wikimedia categories and were not produced by multiple
human annotators. Some photos contain more than one person, and a person may
show more than one pose. The dataset also reflects the photographers, places,
events, and historical material available on Wikimedia Commons. Therefore, it
should not be used for high-stakes decisions or claims about all people and
environments.
