"""
experiments/3_randomized_smoothing_certification.py
Certified Robustness via Randomized Smoothing (Cohen et al.)
Author: MD Hamid Borkot Tulla
"""

import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from scipy.stats import norm
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/certification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

Path('results').mkdir(exist_ok=True)


class RandomizedSmoothingCertifier:
    """
    Certified robustness via randomized smoothing (Cohen et al., ICML 2019)
    Provides L2 robustness certificates
    """
    
    def __init__(self, model, sigma=0.25):
        """
        Args:
            model: Keras/TF model
            sigma: Gaussian noise standard deviation
        """
        self.model = model
        self.sigma = sigma
    
    def certify(self, x, n0=100, n=100000, alpha=0.001):
        """
        Certify robustness of a single sample
        
        Args:
            x: Input sample (1, 42)
            n0: Number of samples for initial prediction
            n: Number of samples for confidence bound
            alpha: Failure probability
        
        Returns:
            (prediction, certified_radius)
        """
        # Step 1: Initial prediction on noisy samples
        batch_size = 1000
        counts_0 = np.zeros(2)  # Binary classification
        
        for _ in range(0, n0, batch_size):
            batch_size_actual = min(batch_size, n0)
            noise = np.random.normal(0, self.sigma, (batch_size_actual, x.shape[1]))
            x_noisy = x + noise
            x_noisy = np.clip(x_noisy, 0, 1)
            
            preds = self.model(x_noisy, training=False).numpy()
            preds_binary = (preds > 0.5).astype(int).squeeze()
            
            for pred in preds_binary:
                counts_0[pred] += 1
        
        prediction = np.argmax(counts_0)
        
        # Step 2: Compute certified radius
        # Count samples agreeing with prediction
        counts_1 = np.zeros(2)
        
        for _ in range(0, n, batch_size):
            batch_size_actual = min(batch_size, n)
            noise = np.random.normal(0, self.sigma, (batch_size_actual, x.shape[1]))
            x_noisy = x + noise
            x_noisy = np.clip(x_noisy, 0, 1)
            
            preds = self.model(x_noisy, training=False).numpy()
            preds_binary = (preds > 0.5).astype(int).squeeze()
            
            for pred in preds_binary:
                counts_1[pred] += 1
        
        # Certified radius computation (Cohen et al. Eq. 5)
        n_A = int(counts_1[prediction])
        n_B = int(counts_1[1 - prediction])
        
        # Confidence bound
        z_alpha = norm.ppf(1 - alpha)
        lower_bound = (n_A - (z_alpha * np.sqrt(n_A) + z_alpha**2 / 2)) / n
        upper_bound = (n_B + (z_alpha * np.sqrt(n_B) + z_alpha**2 / 2)) / n
        
        # Certified radius
        if lower_bound > upper_bound:
            radius = self.sigma * (lower_bound - upper_bound) / 2
        else:
            radius = 0.0
        
        return prediction, float(radius)
    
    def batch_certify(self, X, n0=100, n=100000, alpha=0.001):
        """Certify batch of samples"""
        predictions = []
        radii = []
        
        for i in tqdm(range(len(X)), desc="Certifying samples"):
            x = X[i:i+1]
            pred, radius = self.certify(x, n0=n0, n=n, alpha=alpha)
            predictions.append(int(pred))
            radii.append(radius)
        
        return np.array(predictions), np.array(radii)


def evaluate_certified_robustness(model, X_test, y_test, sigma=0.25, n0=100, n=100000, alpha=0.001):
    """Evaluate certified robustness"""
    logger.info("Evaluating certified robustness via randomized smoothing...")
    logger.info(f"Sigma={sigma}, n0={n0}, n={n}, alpha={alpha}")
    
    certifier = RandomizedSmoothingCertifier(model, sigma=sigma)
    
    # Certify test set (use subset for efficiency)
    subset_size = min(500, len(X_test))
    indices = np.random.choice(len(X_test), subset_size, replace=False)
    X_subset = X_test[indices]
    y_subset = y_test[indices]
    
    predictions, radii = certifier.batch_certify(X_subset, n0=n0, n=n, alpha=alpha)
    
    # Compute certified accuracy at different radii
    certified_accs = {}
    radius_thresholds = [0.1, 0.2, 0.3, 0.42, 0.5]
    
    for r_threshold in radius_thresholds:
        cert_mask = radii >= r_threshold
        if np.sum(cert_mask) > 0:
            certified_acc = np.mean(predictions[cert_mask] == y_subset[cert_mask])
        else:
            certified_acc = 0.0
        
        certified_accs[f'radius_{r_threshold}'] = {
            'certified_accuracy': float(certified_acc),
            'num_certified': int(np.sum(cert_mask)),
            'total': int(len(y_subset))
        }
        logger.info(f"  Radius >= {r_threshold}: {certified_acc:.4f} ({np.sum(cert_mask)}/{len(y_subset)})")
    
    return {
        'predictions': predictions.tolist(),
        'radii': radii.tolist(),
        'certified_accuracies': certified_accs,
        'mean_radius': float(np.mean(radii)),
        'median_radius': float(np.median(radii))
    }


def main(args):
    """Main certification pipeline"""
    
    logger.info("="*60)
    logger.info("Randomized Smoothing Certification")
    logger.info("="*60)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = keras.models.load_model(args.model_path)
    
    # Load test data
    logger.info(f"Loading test data from {args.data_path}")
    X_test = np.load(os.path.join(args.data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_path, 'y_test.npy'))
    
    logger.info(f"Test set: {X_test.shape}")
    
    # Evaluate certified robustness
    results = evaluate_certified_robustness(
        model, X_test, y_test,
        sigma=args.sigma,
        n0=args.n0,
        n=args.n,
        alpha=args.alpha
    )
    
    # Save results
    with open('results/certified_robustness_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to results/certified_robustness_results.json")
    
    # Generate summary
    summary = {
        'configuration': {
            'sigma': args.sigma,
            'n0': args.n0,
            'n': args.n,
            'alpha': args.alpha
        },
        'statistics': {
            'mean_radius': results['mean_radius'],
            'median_radius': results['median_radius'],
            'certified_accuracies': results['certified_accuracies']
        }
    }
    
    with open('results/certification_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Summary saved to results/certification_summary.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Certified robustness evaluation')
    parser.add_argument('--model_path', type=str, default='models/xar_dnn_tf/xar_dnn.h5',
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, default='data/processed/',
                       help='Path to processed test data')
    parser.add_argument('--sigma', type=float, default=0.25,
                       help='Gaussian noise standard deviation')
    parser.add_argument('--n0', type=int, default=100,
                       help='Samples for initial prediction')
    parser.add_argument('--n', type=int, default=100000,
                       help='Samples for confidence bound')
    parser.add_argument('--alpha', type=float, default=0.001,
                       help='Failure probability')
    
    args = parser.parse_args()
    main(args)
