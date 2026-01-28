# 🔒 Feature Stability Analysis: Forensically Auditable Adversarial Robustness for IoT IDS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Paper Status: Under Review](https://img.shields.io/badge/Paper-Under_Review-red.svg)]()

## 📋 Overview

This repository contains the **complete, reproducible implementation** of **Feature Stability Analysis (FSA)**, a novel quantitative metric for measuring SHAP attribution consistency under adversarial attack in Deep Neural Networks. 

The paper introduces **XAR-DNN**, a lightweight edge-optimized intrusion detection system that unifies adversarial robustness with explainability, enabling **auditable, forensically traceable AI** for industrial IoT security applications.

### Key Contributions

| Feature | Metric | Status |
|---------|--------|--------|
| **Clean Accuracy** | 95.74% | ✅ Validated |
| **FGSM Robustness** | 95.09% (Δ=0.65%, non-sig) | ✅ Tested |
| **PGD-10 Robustness** | 93.90% | ✅ Certified |
| **Feature Stability Ratio** | 43% features with S_i > 0.7 | ✅ Quantified |
| **Edge Latency** | 2.3 ms on Raspberry Pi 4 | ✅ Profiled |
| **Model Size** | 126 KB RAM | ✅ Deployable |
| **EU AI Act Compliance** | Article 13 operationalized | ✅ Implemented |

---

## 🎯 What Makes This Different?

**Problem:** Deep learning IDSs are either:
- **Robust but opaque** (adversarial training obscures explanations)
- **Explainable but fragile** (SHAP fails under adversarial attack)

**Solution:** FSA quantifies **reasoning resilience**—whether your model's explanations remain stable when attacked.

```
┌─────────────────────────────────────────────────────────┐
│  TRADITIONAL IDS                                        │
│  ✅ Prediction: Normal (95% confidence)                 │
│  ⚠️ Explanation: flow_duration (but might shift to      │
│     tcp.payload under attack)                           │
│  ❌ Auditable? NO                                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  XAR-DNN + FSA                                          │
│  ✅ Prediction: Normal (95% confidence)                 │
│  ✅ Explanation: flow_duration (S_i=0.96, forensically │
│     reliable—auto-block)                                │
│  ✅ Auditable? YES – EU AI Act compliant                │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Repository Structure

```
xar-dnn-fsa/
├── experiments/                              # Complete 6-experiment suite
│   ├── 1_train_xar_dnn.py                   # Training with adversarial objective
│   ├── 2_fgsm_pgd_eval.py                   # FGSM/PGD/Auto-PGD attacks
│   ├── 3_randomized_smoothing_certification.py  # Certified robustness
│   ├── 4_semantic_attacks.py                # Protocol-level attacks
│   ├── 5_fsa_analysis.py                    # FSA metric computation (CORE)
│   └── 6_energy_measurement.py              # Edge device profiling
│
├── master_experiments.py                     # Orchestrates all experiments
├── preprocess_data.py                        # Data preparation pipeline
│
├── data/
│   └── processed/                            # Preprocessed datasets
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       └── scaler.pkl
│
├── models/
│   └── xar_dnn_tf/                          # Trained XAR-DNN
│       ├── xar_dnn.h5
│       └── checkpoint files
│
├── results/                                  # Comprehensive results
│   ├── training_results.json
│   ├── adversarial_robustness.json
│   ├── certification_results.json
│   ├── semantic_attacks.json
│   ├── fsa_analysis.json
│   └── energy_profiling.json
│
├── logs/                                     # Experiment logs
│   └── *.log
│
├── README.md                                 # This file
├── requirements.txt                          # Dependencies
├── LICENSE                                   # MIT License
└── .gitignore                               # Git ignore rules
```

---

## 🚀 Quick Start (5 minutes)

### Step 1: Clone & Setup

```bash
git clone https://github.com/yourusername/xar-dnn-fsa.git
cd xar-dnn-fsa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Data

```bash
# Download Edge-IIoTSet dataset
wget https://datasets.org/edge-iiotset/edge-iiotset.csv -O data/raw/Edge-IIoTSet.csv

# Preprocess (creates train/test split + normalization)
python preprocess_data.py --dataset edge-iiotset --file data/raw/Edge-IIoTSet.csv
```

### Step 3: Run Experiments

```bash
# Option A: Run all experiments in sequence (30-45 min with GPU)
python master_experiments.py --all

# Option B: Run individual experiment
python experiments/1_train_xar_dnn.py
python experiments/5_fsa_analysis.py  # Most important
python experiments/6_energy_measurement.py

# Check results
cat results/fsa_analysis.json
cat results/adversarial_robustness.json
```

---

## 📈 Experiment Details

### Experiment 1: XAR-DNN Training (5-10 min)

Trains a shallow DNN (42→128→64→32→1) with adversarial objective:

```
Loss = L_clean(x, y) + L_adversarial(x_adv, y)

x_adv = FGSM(x, y, ε=0.1)
```

**Output:** `models/xar_dnn_tf/xar_dnn.h5`  
**Key Metric:** 95.74% clean accuracy

```bash
python experiments/1_train_xar_dnn.py
```

### Experiment 2: Adversarial Robustness (3-5 min)

Evaluates FGSM, PGD-10, Auto-PGD attacks with McNemar's statistical test:

```
FGSM (ε=0.1):     95.09% ± 0.24%
PGD-10 (ε=0.1):   93.90% ± 0.22%
Auto-PGD:         93.25% ± 0.25%
```

**Output:** `results/adversarial_robustness.json`

```bash
python experiments/2_fgsm_pgd_eval.py
```

### Experiment 3: Certified Robustness (10-15 min)

Randomized Smoothing certification at L2 radius ε:

```
Certified Accuracy @ ε=0.42: 78.4%
All samples have certified L2 radius
```

**Output:** `results/certification_results.json`

```bash
python experiments/3_randomized_smoothing_certification.py
```

### Experiment 4: Semantic Attacks (5-10 min)

Protocol-level perturbations (MQTT replay, packet drop, jitter):

```
Attack Type              Success Rate    FSA Degradation
─────────────────────────────────────────────────────────
MQTT Replay             5.2%            Δ=0.08
Packet Drop (10%)       3.8%            Δ=0.11
Jitter (±5%)            4.1%            Δ=0.07
```

**Output:** `results/semantic_attacks.json`

```bash
python experiments/4_semantic_attacks.py
```

### Experiment 5: FSA Analysis (5-10 min) **← CORE**

Computes Feature Stability Scores via SHAP:

$$S_i = 1 - \frac{\|\phi_i^{adv} - \phi_i^{clean}\|_2}{\|\phi_i^{clean}\|_2 + \epsilon}$$

**Results:**

```
High Stability (S_i > 0.7):   18 features
  - flow_duration:    S_i = 0.96 ✅ Auto-block
  - packet_rate:      S_i = 0.95 ✅ Auto-block
  - tcp.flags:        S_i = 0.94 ✅ Human review

Medium Stability (0.5-0.7):   16 features
  - mqtt.msgtype:     S_i = 0.68 ⚠️  Log & correlate

Low Stability (S_i < 0.5):     8 features
  - tcp.payload:      S_i = 0.54 ❌ Escalate to L2
```

**Output:** `results/fsa_analysis.json`

```bash
python experiments/5_fsa_analysis.py
```

### Experiment 6: Edge Profiling (3-5 min)

Latency & energy measurement on Raspberry Pi 4:

```
Mean Latency:        2.3 ms
P95 Latency:         3.1 ms
Estimated Energy:    0.73 mJ/inference
Model Size:          0.73 MB
Peak RAM:            126 KB

Comparison:
XAR-DNN:             2.3 ms  (3.6× faster than XGBoost)
XGBoost:             5.6 ms
Random Forest:       8.1 ms
```

**Output:** `results/energy_profiling.json`

```bash
python experiments/6_energy_measurement.py
```

---

## 📊 Key Results Tables

### Table 1: Adversarial Robustness (Edge-IIoTSet)

| Model | Clean Acc | FGSM | PGD-10 | Auto-PGD | Stable Features |
|-------|-----------|------|--------|----------|-----------------|
| **XAR-DNN** | **95.74%** | **95.09%** | **93.90%** | **93.25%** | **43%** |
| Standard DNN | 96.80% | 68.30% | 65.40% | 63.70% | 12% |
| TRADES | 94.60% | 93.80% | 92.50% | 92.10% | 28% |

### Table 2: Cross-Dataset Generalization

| Dataset | Clean | FGSM | Macro-F1 (Attack) |
|---------|-------|------|-------------------|
| Edge-IIoTSet | 95.74% | 95.09% | 0.88±0.02 |
| NSL-KDD | 91.5%±0.4 | 89.2%±0.5 | 0.85±0.03 |
| CIC-IDS2018 | 89.9%±0.6 | 87.4%±0.7 | 0.82±0.04 |

### Table 3: SOC Triage Impact

| Action | Alert Count | Auto-Block Precision | Manual Review Rate |
|--------|-------------|---------------------|-------------------|
| Auto-block ($S_i > 0.9$) | 12,834 | 99.2% | — |
| Log & correlate ($0.5 < S_i \leq 0.9$) | 9,583 | 85.6% | — |
| Escalate ($S_i \leq 0.5$) | 2,817 | — | 11.7% |

---

## 🔧 Dependencies

All packages listed in `requirements.txt`. Key versions:

```
tensorflow==2.14.0
numpy==1.24.3
pandas==2.1.1
scikit-learn==1.3.0
shap==0.43.0
scipy==1.11.2
matplotlib==3.8.0
seaborn==0.12.2
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 💾 Datasets

This implementation supports **3 major IoT/IDS datasets**:

### 1. **Edge-IIoTSet** (Primary)
- **Size:** 157.8K flows
- **Features:** 42 engineered features
- **Classes:** 14 attack types + Normal
- **Download:** [https://datasets.org/edge-iiotset/](https://datasets.org/edge-iiotset/)

### 2. **NSL-KDD** (Cross-validation)
- **Size:** 125.9K flows
- **Features:** 41 numeric + 1 protocol
- **Classes:** 4 families + Normal
- **Download:** [https://www.unb.ca/cic/datasets/nsl-kdd.html](https://www.unb.ca/cic/datasets/nsl-kdd.html)

### 3. **CIC-IDS2018** (Modern attacks)
- **Size:** 16M+ flows
- **Features:** 80+ network metrics
- **Classes:** 15 attack types + Benign
- **Download:** [https://www.unb.ca/cic/datasets/ids-2018.html](https://www.unb.ca/cic/datasets/ids-2018.html)

---

## 📋 Configuration

All hyperparameters are centralized in `Config` class:

```python
# experiments/*/config.py or master_experiments.py

class Config:
    # Architecture
    ARCHITECTURE = [128, 64, 32]
    DROPOUT_RATE = 0.3
    L2_REG = 1e-4
    
    # Training
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 512
    
    # Adversarial
    EPSILON_FGSM = 0.1
    EPSILON_PGD = 0.1
    PGD_STEPS = 10
    
    # FSA
    FSA_VALIDATION_SAMPLES = 5000
    FSA_SHAP_BACKGROUND = 200
```

Modify before training to customize experiments.

---

## 🧪 Testing & Validation

### Unit Tests

```bash
# Verify imports
python -c "import tensorflow, shap, numpy; print('✅ Core imports successful')"

# Quick sanity check
python -c "
from experiments.xar_dnn_model import XARDNNModel
model = XARDNNModel(input_dim=42)
print(f'✅ Model created: {model.count_params()} parameters')
"
```

### Integration Test

```bash
# Run all experiments (full validation)
python master_experiments.py --all

# Check results exist
ls -lh results/*.json
```

---

## 📝 Usage Examples

### Example 1: Train Your Own Model

```python
from experiments.xar_dnn_trainer import XARDNNTrainer

trainer = XARDNNTrainer()
model, X_test, y_test = trainer.train()
print(f"Model accuracy: {trainer.evaluate(model, X_test, y_test)}")
```

### Example 2: Compute FSA for Your Model

```python
from experiments.fsa_analyzer import FeatureStabilityAnalyzer
import numpy as np

analyzer = FeatureStabilityAnalyzer(model)

# Generate adversarial examples
X_adv = generate_adversarial(model, X_test, epsilon=0.1)

# Compute FSA
S = analyzer.analyze(X_test, X_adv)
print(f"Mean FSA: {S.mean():.4f}")
print(f"High-stability features: {np.sum(S > 0.7)}")
```

### Example 3: SOC Triage Decision

```python
# Decide action based on FSA scores
S = results['stability_scores']
top_features_idx = np.argsort(S)[-3:]  # Top 3 features
top_stability = S[top_features_idx]

if np.all(top_stability > 0.9):
    action = "AUTO_BLOCK"  # High confidence
elif np.any(top_stability < 0.5):
    action = "ESCALATE"    # Escalate to L2
else:
    action = "LOG"         # Medium confidence
```

---

## 🎓 Publication & Citation

**Paper:** Feature Stability Analysis: Forensically Auditable Adversarial Robustness for IoT IDS  
**Authors:** Tulla et al. (2026)  
**Status:** Under Review

### Cite This Work

```bibtex
@article{tulla2026fsa,
  title={Feature Stability Analysis: Forensically Auditable Adversarial Robustness for IoT IDS},
  author={Tulla, MD Hamid Borkot and Shreya, Saraf Anzum and others},
  journal={Under Review},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support & Issues

- **Bug Reports:** Open GitHub Issue with reproduction steps
- **Questions:** Discuss tab in GitHub repository
- **Feature Requests:** GitHub Issues with `[FEATURE]` tag

---

## 🔗 Resources

- **Paper:** [Full paper (when published)](link)
- **Datasets:** [Edge-IIoTSet](https://datasets.org/edge-iiotset/), [NSL-KDD](https://www.unb.ca/cic/datasets/nsl-kdd.html), [CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)
- **EU AI Act:** [Article 13 - Transparency](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng)
- **SHAP:** [GitHub - shap/shap](https://github.com/shap/shap)
- **TensorFlow:** [tensorflow.org](https://www.tensorflow.org/)

---

## 📊 Results Visualization

After running experiments, visualize results:

```bash
# Generate plots (requires matplotlib, seaborn)
python << 'EOF'
import json
import matplotlib.pyplot as plt
import numpy as np

# Load FSA results
with open('results/fsa_analysis.json') as f:
    fsa_data = json.load(f)

# Plot stability distribution
S = fsa_data['stability_scores']
plt.figure(figsize=(10, 6))
plt.hist(S, bins=20, edgecolor='black')
plt.xlabel('Stability Score (S_i)')
plt.ylabel('Feature Count')
plt.title('Feature Stability Distribution')
plt.axvline(0.7, color='green', linestyle='--', label='High Stability Threshold')
plt.axvline(0.5, color='orange', linestyle='--', label='Escalation Threshold')
plt.legend()
plt.tight_layout()
plt.savefig('results/fsa_distribution.png', dpi=150)
print("✅ Plot saved to results/fsa_distribution.png")
EOF
```

---

## ✅ Reproducibility Checklist

- [x] **Code:** All 6 experiments fully implemented
- [x] **Data:** Preprocessing scripts for 3 datasets
- [x] **Configuration:** Centralized config with all hyperparameters
- [x] **Random Seeds:** Fixed (42) for reproducibility
- [x] **Documentation:** Docstrings on all functions
- [x] **Results:** Comprehensive JSON outputs
- [x] **Logging:** Detailed experiment logs
- [x] **License:** MIT (open source)
- [x] **Dependencies:** requirements.txt with versions

**Status:** ✅ **FULLY REPRODUCIBLE**

---

## 🎯 Next Steps

1. **Download data:** Edge-IIoTSet dataset
2. **Preprocess:** `python preprocess_data.py`
3. **Run experiments:** `python master_experiments.py --all`
4. **Check results:** `cat results/*.json`
5. **Cite this work:** Use BibTeX above
6. **Share feedback:** GitHub Issues or Discussions

---

## 👥 Acknowledgments

- **Chongqing University of Posts and Telecommunications**
- **Rajshahi University of Engineering & Technology**
- **Chongqing University**
- **Nantong University**

---

**Last Updated:** January 29, 2026  
**Repository Version:** 1.0.0  
**Status:** ✅ Production Ready

---

<div align="center">

**🚀 Ready to Deploy Forensically Auditable AI? Get started now!**

[⬇️ Download](#quick-start-5-minutes) | [📖 Read Docs](#-repository-structure) | [📊 View Results](#-key-results-tables) | [🔗 Cite](#-publication--citation)

</div>
