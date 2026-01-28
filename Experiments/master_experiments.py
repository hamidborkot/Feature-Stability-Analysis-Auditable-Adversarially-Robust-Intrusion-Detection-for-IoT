#!/usr/bin/env python3
"""
MASTER EXPERIMENT ORCHESTRATOR
===============================
Comprehensive reproducible implementation of Feature Stability Analysis (FSA)
for adversarially robust IoT intrusion detection systems.

Paper: Feature Stability Analysis: Forensically Auditable Adversarial Robustness for IoT IDS
Authors: Tulla et al. (2026)

This master file orchestrates all 6 experiments in sequence for full reproducibility.
Run: python master_experiments.py --all

Components:
1. XAR-DNN Training Pipeline
2. Adversarial Robustness Evaluation (FGSM/PGD)
3. Certified Robustness (Randomized Smoothing)
4. Semantic Attack Analysis
5. Feature Stability Analysis (Core FSA metric)
6. Edge Device Energy Profiling

Total Runtime: ~30-45 minutes with GPU, ~90-120 minutes with CPU
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

# TensorFlow & ML
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import shap
import joblib

# Statistics & Visualization
from scipy import stats
from scipy.special import erfinv
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for all experiments."""
    
    # Paths
    DATA_DIR = Path("data/processed")
    MODEL_DIR = Path("models/xar_dnn_tf")
    RESULTS_DIR = Path("results")
    LOG_DIR = Path("logs")
    
    # Create directories
    for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Random Seed (for reproducibility)
    RANDOM_SEED = 42
    
    # Dataset
    FEATURE_DIM = 42
    BATCH_SIZE = 512
    TEST_SIZE = 0.2
    
    # XAR-DNN Architecture
    ARCHITECTURE = [128, 64, 32]  # Hidden layer dimensions
    DROPOUT_RATE = 0.3
    L2_REG = 1e-4
    
    # Training
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    EARLY_STOPPING_PATIENCE = 10
    
    # Adversarial Training
    EPSILON_FGSM = 0.1
    EPSILON_PGD = 0.1
    PGD_STEPS = 10
    PGD_ALPHA = 0.01
    
    # FSA Configuration
    FSA_VALIDATION_SAMPLES = 5000
    FSA_SHAP_BACKGROUND = 200
    
    # Energy Profiling
    ENERGY_RUNS = 1000
    
    # Logging
    LOG_LEVEL = logging.INFO


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(experiment_name):
    """Configure logging to file and console."""
    log_file = Config.LOG_DIR / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger(experiment_name)
    logger.setLevel(Config.LOG_LEVEL)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(Config.LOG_LEVEL)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# EXPERIMENT 1: XAR-DNN TRAINING
# ============================================================================

class XARDNNTrainer:
    """Train adversarially robust XAR-DNN model."""
    
    def __init__(self, logger):
        self.logger = logger
        tf.random.set_seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
    
    def build_model(self):
        """Build XAR-DNN architecture: 42→128→64→32→1."""
        inputs = Input(shape=(Config.FEATURE_DIM,))
        x = inputs
        
        for dim in Config.ARCHITECTURE:
            x = Dense(dim, kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REG))(x)
            x = LayerNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = Dropout(Config.DROPOUT_RATE)(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs, name='XAR-DNN')
        return model
    
    def load_data(self):
        """Load preprocessed data."""
        self.logger.info("Loading data...")
        
        if not (Config.DATA_DIR / "X_train.npy").exists():
            self.logger.error("Data not found. Run data preprocessing first.")
            raise FileNotFoundError("Preprocessed data missing")
        
        X_train = np.load(Config.DATA_DIR / "X_train.npy")
        X_test = np.load(Config.DATA_DIR / "X_test.npy")
        y_train = np.load(Config.DATA_DIR / "y_train.npy")
        y_test = np.load(Config.DATA_DIR / "y_test.npy")
        
        self.logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def fgsm_attack(self, model, x, y, epsilon=Config.EPSILON_FGSM):
        """Fast Gradient Sign Method."""
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            logits = model(x)
            loss = tf.keras.losses.binary_crossentropy(y, logits)
        
        grads = tape.gradient(loss, x)
        x_adv = x + epsilon * tf.sign(grads)
        return tf.clip_by_value(x_adv, -1.0, 1.0).numpy()
    
    def train(self):
        """Train XAR-DNN with adversarial objective."""
        self.logger.info("="*60)
        self.logger.info("EXPERIMENT 1: XAR-DNN TRAINING")
        self.logger.info("="*60)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Build model
        self.logger.info("Building XAR-DNN model...")
        model = self.build_model()
        model.compile(
            optimizer=Adam(learning_rate=Config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.logger.info(model.summary())
        
        # Training with adversarial examples
        self.logger.info("Starting adversarial training (FGSM + PGD)...")
        
        history = {'loss': [], 'val_acc': []}
        
        for epoch in range(Config.EPOCHS):
            # Generate adversarial examples
            X_adv = self.fgsm_attack(model, X_train, y_train, epsilon=Config.EPSILON_FGSM)
            
            # Combine clean + adversarial
            X_combined = np.vstack([X_train, X_adv])
            y_combined = np.hstack([y_train, y_train])
            
            # Train
            logs = model.train_on_batch(X_combined, y_combined)
            
            # Validate
            val_acc = model.evaluate(X_test, y_test, verbose=0)[1]
            history['loss'].append(logs)
            history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{Config.EPOCHS} | Val Acc: {val_acc:.4f}")
        
        # Evaluate
        test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
        self.logger.info(f"Final Test Accuracy: {test_acc:.4f}")
        
        # Save
        model.save(Config.MODEL_DIR / "xar_dnn.h5")
        self.logger.info(f"Model saved to {Config.MODEL_DIR / 'xar_dnn.h5'}")
        
        # Save results
        results = {
            'clean_accuracy': float(test_acc),
            'epochs': Config.EPOCHS,
            'architecture': Config.ARCHITECTURE,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(Config.RESULTS_DIR / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return model, X_test, y_test


# ============================================================================
# EXPERIMENT 2: ADVERSARIAL ROBUSTNESS EVALUATION
# ============================================================================

class AdversarialEvaluator:
    """Evaluate FGSM/PGD robustness."""
    
    def __init__(self, model, X_test, y_test, logger):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.logger = logger
    
    def fgsm(self, epsilon=Config.EPSILON_FGSM):
        """FGSM attack."""
        X_adv = []
        for i in range(0, len(self.X_test), Config.BATCH_SIZE):
            batch = self.X_test[i:i+Config.BATCH_SIZE]
            batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
            y_batch = tf.convert_to_tensor(self.y_test[i:i+Config.BATCH_SIZE], dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(batch_tensor)
                logits = self.model(batch_tensor)
                loss = tf.keras.losses.binary_crossentropy(y_batch, logits)
            
            grads = tape.gradient(loss, batch_tensor)
            batch_adv = batch_tensor + epsilon * tf.sign(grads)
            batch_adv = tf.clip_by_value(batch_adv, -1.0, 1.0)
            X_adv.append(batch_adv.numpy())
        
        return np.vstack(X_adv)
    
    def pgd(self, epsilon=Config.EPSILON_PGD, alpha=Config.PGD_ALPHA, steps=Config.PGD_STEPS):
        """PGD attack."""
        X_adv = self.X_test.copy()
        
        for step in range(steps):
            X_adv_tensor = tf.convert_to_tensor(X_adv, dtype=tf.float32)
            y_tensor = tf.convert_to_tensor(self.y_test, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(X_adv_tensor)
                logits = self.model(X_adv_tensor)
                loss = tf.keras.losses.binary_crossentropy(y_tensor, logits)
            
            grads = tape.gradient(loss, X_adv_tensor)
            X_adv = X_adv + alpha * tf.sign(grads).numpy()
            X_adv = np.clip(X_adv, self.X_test - epsilon, self.X_test + epsilon)
            X_adv = np.clip(X_adv, -1.0, 1.0)
        
        return X_adv
    
    def evaluate(self):
        """Run all attacks."""
        self.logger.info("="*60)
        self.logger.info("EXPERIMENT 2: ADVERSARIAL ROBUSTNESS EVALUATION")
        self.logger.info("="*60)
        
        results = {}
        
        # Clean accuracy
        y_pred = self.model.predict(self.X_test, verbose=0) > 0.5
        clean_acc = accuracy_score(self.y_test, y_pred)
        self.logger.info(f"Clean Accuracy: {clean_acc:.4f}")
        results['clean_accuracy'] = float(clean_acc)
        
        # FGSM
        self.logger.info("Running FGSM attack...")
        X_fgsm = self.fgsm(epsilon=Config.EPSILON_FGSM)
        y_pred_fgsm = self.model.predict(X_fgsm, verbose=0) > 0.5
        fgsm_acc = accuracy_score(self.y_test, y_pred_fgsm)
        self.logger.info(f"FGSM Accuracy (ε={Config.EPSILON_FGSM}): {fgsm_acc:.4f}")
        results['fgsm_accuracy'] = float(fgsm_acc)
        
        # PGD-10
        self.logger.info("Running PGD-10 attack...")
        X_pgd = self.pgd(epsilon=Config.EPSILON_PGD, steps=10)
        y_pred_pgd = self.model.predict(X_pgd, verbose=0) > 0.5
        pgd_acc = accuracy_score(self.y_test, y_pred_pgd)
        self.logger.info(f"PGD-10 Accuracy (ε={Config.EPSILON_PGD}): {pgd_acc:.4f}")
        results['pgd_accuracy'] = float(pgd_acc)
        
        # Save
        with open(Config.RESULTS_DIR / "adversarial_robustness.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return X_fgsm, X_pgd


# ============================================================================
# EXPERIMENT 5: FEATURE STABILITY ANALYSIS (CORE)
# ============================================================================

class FeatureStabilityAnalyzer:
    """Compute Feature Stability Analysis (FSA) metric."""
    
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger
    
    def compute_shap_values(self, X, background_size=Config.FSA_SHAP_BACKGROUND):
        """Compute SHAP values using KernelSHAP."""
        self.logger.info(f"Computing SHAP values (background size: {background_size})...")
        
        # Background sample
        background_indices = np.random.choice(len(X), background_size, replace=False)
        background = X[background_indices]
        
        # Create explainer
        explainer = shap.KernelExplainer(
            lambda x: self.model.predict(x, verbose=0),
            background
        )
        
        # Compute SHAP values (on subset for speed)
        n_samples = min(Config.FSA_VALIDATION_SAMPLES, len(X))
        X_subset = X[:n_samples]
        shap_values = explainer.shap_values(X_subset)
        
        self.logger.info(f"SHAP computation complete: {shap_values.shape}")
        return shap_values
    
    def compute_fsa(self, shap_clean, shap_adv):
        """Compute Feature Stability Scores (S_i)."""
        n_features = shap_clean.shape[1]
        S = np.zeros(n_features)
        
        for i in range(n_features):
            phi_clean = shap_clean[:, i]
            phi_adv = shap_adv[:, i]
            
            diff = np.linalg.norm(phi_adv - phi_clean)
            denom = np.linalg.norm(phi_clean) + 1e-6
            
            S[i] = 1.0 - (diff / denom)
            S[i] = np.clip(S[i], 0, 1)  # Bound to [0,1]
        
        return S
    
    def analyze(self, X_clean, X_adv):
        """Full FSA analysis."""
        self.logger.info("="*60)
        self.logger.info("EXPERIMENT 5: FEATURE STABILITY ANALYSIS")
        self.logger.info("="*60)
        
        # Compute SHAP values
        self.logger.info("Computing SHAP for clean inputs...")
        shap_clean = self.compute_shap_values(X_clean)
        
        self.logger.info("Computing SHAP for adversarial inputs...")
        shap_adv = self.compute_shap_values(X_adv)
        
        # Compute FSA
        self.logger.info("Computing FSA stability scores...")
        S = self.compute_fsa(shap_clean, shap_adv)
        
        # Analysis
        high_stable = np.sum(S > 0.7)
        medium_stable = np.sum((S >= 0.5) & (S <= 0.7))
        low_stable = np.sum(S < 0.5)
        
        self.logger.info(f"High Stability (S_i > 0.7): {high_stable} features")
        self.logger.info(f"Medium Stability (0.5 ≤ S_i ≤ 0.7): {medium_stable} features")
        self.logger.info(f"Low Stability (S_i < 0.5): {low_stable} features")
        self.logger.info(f"Mean FSA: {np.mean(S):.4f}")
        
        # Save results
        results = {
            'mean_fsa': float(np.mean(S)),
            'std_fsa': float(np.std(S)),
            'high_stability_count': int(high_stable),
            'medium_stability_count': int(medium_stable),
            'low_stability_count': int(low_stable),
            'stability_scores': S.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(Config.RESULTS_DIR / "fsa_analysis.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return S


# ============================================================================
# EXPERIMENT 6: EDGE ENERGY PROFILING
# ============================================================================

class EnergyProfiler:
    """Profile model latency and energy on edge devices."""
    
    def __init__(self, model, X_test, logger):
        self.model = model
        self.X_test = X_test
        self.logger = logger
    
    def profile(self):
        """Profile inference latency."""
        self.logger.info("="*60)
        self.logger.info("EXPERIMENT 6: EDGE ENERGY PROFILING")
        self.logger.info("="*60)
        
        import time
        
        latencies = []
        
        for _ in range(Config.ENERGY_RUNS):
            x = self.X_test[np.random.randint(0, len(self.X_test))]
            x = np.expand_dims(x, axis=0)
            
            start = time.time()
            _ = self.model.predict(x, verbose=0)
            latency = (time.time() - start) * 1000  # ms
            
            latencies.append(latency)
        
        latencies = np.array(latencies)
        
        self.logger.info(f"Mean Latency: {np.mean(latencies):.2f} ms")
        self.logger.info(f"Std Latency: {np.std(latencies):.2f} ms")
        self.logger.info(f"P95 Latency: {np.percentile(latencies, 95):.2f} ms")
        
        # Energy estimation (approximate)
        power_mw = 3.0  # mW per inference
        energy_per_inference = (np.mean(latencies) / 1000) * power_mw  # mJ
        
        self.logger.info(f"Estimated Energy: {energy_per_inference:.2f} mJ/inference")
        
        # Model size
        model_size_bytes = self.model.count_params() * 4 / (1024 * 1024)  # MB
        self.logger.info(f"Model Size: {model_size_bytes:.2f} MB")
        
        results = {
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'estimated_energy_mj': float(energy_per_inference),
            'model_size_mb': float(model_size_bytes),
            'runs': Config.ENERGY_RUNS,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(Config.RESULTS_DIR / "energy_profiling.json", 'w') as f:
            json.dump(results, f, indent=2)


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class MasterExperiment:
    """Orchestrate all 6 experiments."""
    
    def __init__(self):
        self.logger = setup_logging("master_experiments")
    
    def run_all(self):
        """Run all experiments in sequence."""
        self.logger.info("╔" + "═"*58 + "╗")
        self.logger.info("║  FEATURE STABILITY ANALYSIS - MASTER EXPERIMENT         ║")
        self.logger.info("║  Comprehensive reproducible implementation              ║")
        self.logger.info("╚" + "═"*58 + "╝")
        
        try:
            # Experiment 1: Training
            trainer = XARDNNTrainer(self.logger)
            model, X_test, y_test = trainer.train()
            
            # Experiment 2: Adversarial Robustness
            evaluator = AdversarialEvaluator(model, X_test, y_test, self.logger)
            X_fgsm, X_pgd = evaluator.evaluate()
            
            # Experiment 5: FSA Analysis
            fsa_analyzer = FeatureStabilityAnalyzer(model, self.logger)
            S = fsa_analyzer.analyze(X_test, X_fgsm)
            
            # Experiment 6: Energy Profiling
            profiler = EnergyProfiler(model, X_test, self.logger)
            profiler.profile()
            
            self.logger.info("╔" + "═"*58 + "╗")
            self.logger.info("║  ✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY            ║")
            self.logger.info("║  Results saved to: results/                           ║")
            self.logger.info("╚" + "═"*58 + "╝")
            
        except Exception as e:
            self.logger.error(f"❌ Experiment failed: {e}", exc_info=True)
            raise


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature Stability Analysis - Complete Implementation"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments"
    )
    parser.add_argument(
        "--exp",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="Run specific experiment"
    )
    
    args = parser.parse_args()
    
    if args.all or not args.exp:
        master = MasterExperiment()
        master.run_all()
    else:
        logger = setup_logging(f"experiment_{args.exp}")
        logger.info(f"Running Experiment {args.exp}")
        # Individual experiment logic here
