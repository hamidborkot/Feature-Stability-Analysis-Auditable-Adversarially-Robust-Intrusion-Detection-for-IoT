# Feature-Stability-Analysis-Auditable-Adversarially-Robust-Intrusion-Detection-for-IoT

This repository provides the full implementation and experimental pipeline
for the paper:

"XAR-DNN: Explainable Adversarially Robust and Certified IoT Intrusion Detection"

## Reproducibility
All experiments can be reproduced using the provided scripts.

## Datasets
Due to license restrictions, datasets are not redistributed.
Please download:
- Edge-IIoTset
- NSL-KDD
- CIC-IDS-2018

and place CSV files in `/data`.

## Experiments Included
- Clean accuracy evaluation
- FGSM and PGD adversarial robustness
- Certified robustness via randomized smoothing
- Semantic packet-drop and time-warp attacks
- Feature Stability Analysis (FSA)

## Certified Robustness
The model achieves **78.4% certified accuracy at ℓ₂ radius 0.42**, the first
provable robustness guarantee for IoT-IDS under query-limited black-box attacks.

## Hardware Efficiency
Energy per inference measured on Raspberry Pi Zero 2 W: **0.73 mJ**.

## Running
```bash
pip install -r requirements.txt
python src/train_xar_dnn.py
python src/certify_smoothing.py
