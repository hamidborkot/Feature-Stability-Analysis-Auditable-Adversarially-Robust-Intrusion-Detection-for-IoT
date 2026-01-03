# Feature-Stability-Analysis-Auditable-Adversarially-Robust-Intrusion-Detection-for-IoT

This repository provides the full implementation and experimental pipeline
for the paper:

"XAR-DNN: Explainable Adversarially Robust and Certified IoT Intrusion Detection"

## Reproducibility
All experiments can be reproduced using the provided scripts.

## Datasets
Due to license restrictions, datasets are not redistributed.
Please download:
- Edge-IIoTset -https://www.kaggle.com/code/mdhamidborkottulla/notebook304f1cf782/edit
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

---

## 🧪 Experiment–Paper Mapping

| Paper Section | Script |
|--------------|--------|
| IV-B Model Training | `01_train_xar_dnn.py` |
| IV-C FGSM / PGD Robustness | `02_fgsm_pgd_eval.py` |
| IV-D Certified ℓ₂ Robustness | `03_randomized_smoothing_certification.py` |
| IV-E Semantic Attacks | `04_semantic_attacks.py` |
| IV-F Feature Stability (FSA) | `05_fsa_analysis.py` |
| IV-G Energy Efficiency | `06_energy_measurement.py` |

---

## 🔐 Key Results Summary

| Metric | Value |
|------|------|
| Clean Accuracy | **95.74 %** |
| FGSM Accuracy | **95.09 %** |
| PGD-5 Accuracy | **94.80 %** |
| Certified Acc @ ℓ₂ = 0.42 | **78.4 %** |
| ROC-AUC | **0.97** |
| Energy / Inference | **0.73 mJ** |

---

## 📊 Interactive Results Panel

See **`docs/results.html`** for a unified, visual summary of all results.

---

## 📦 Datasets

Due to size restrictions, datasets are not included:
- Edge-IIoTset
- NSL-KDD
- CIC-IDS-2018

Official download links are provided in `data/README.md`.

---

## 🧾 Reproducibility

All experiments were conducted with:
- Fixed random seeds
- Explicit hyperparameters
- Hardware and software versions disclosed

---

## 📜 License

This project is released for **academic research purposes**.

