"""
experiments/2_fgsm_pgd_eval.py
Complete FGSM and PGD Adversarial Evaluation
Author: MD Hamid Borkot Tulla
"""

import os
import json
import pickle
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
import joblib
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/adversarial_eval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

Path('results').mkdir(exist_ok=True)
Path('results/adversarial_examples').mkdir(exist_ok=True)


class AdversarialEvaluator:
    """Compute adversarial perturbations and evaluate robustness"""
    
    def __init__(self, model):
        self.model = model
        self.loss_fn = keras.losses.BinaryCrossentropy()
    
    def fgsm_attack(self, x, y, epsilon=0.1):
        """Fast Gradient Sign Method - single step attack"""
        x_tensor = tf.Variable(x, trainable=True, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            logits = self.model(x_tensor, training=False)
            loss = self.loss_fn(y, logits)
        
        grads = tape.gradient(loss, x_tensor)
        x_adv = x + epsilon * tf.sign(grads)
        x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)
        
        return x_adv.numpy()
    
    def pgd_attack(self, x, y, epsilon=0.1, alpha=0.01, num_steps=10):
        """Projected Gradient Descent - iterative attack"""
        x_adv = x + np.random.uniform(-epsilon, epsilon, x.shape).astype(np.float32)
        x_adv = np.clip(x_adv, 0, 1)
        
        for step in range(num_steps):
            x_tensor = tf.Variable(x_adv, trainable=True, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                logits = self.model(x_tensor, training=False)
                loss = self.loss_fn(y, logits)
            
            grads = tape.gradient(loss, x_tensor)
            x_adv = x_adv + alpha * np.sign(grads.numpy())
            x_adv = np.clip(x_adv, x - epsilon, x + epsilon)
            x_adv = np.clip(x_adv, 0, 1)
        
        return x_adv
    
    def auto_pgd_attack(self, x, y, epsilon=0.1, num_steps=20):
        """AutoAttack's PGD variant with adaptive step size"""
        alpha = epsilon / num_steps * 2.5
        x_adv = x + np.random.uniform(-epsilon, epsilon, x.shape).astype(np.float32)
        x_adv = np.clip(x_adv, 0, 1)
        
        for step in range(num_steps):
            x_tensor = tf.Variable(x_adv, trainable=True, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                logits = self.model(x_tensor, training=False)
                loss = self.loss_fn(y, logits)
            
            grads = tape.gradient(loss, x_tensor)
            
            # Adaptive step size
            grad_norm = np.linalg.norm(grads.numpy().flatten())
            if grad_norm > 0:
                x_adv = x_adv + (alpha / grad_norm) * grads.numpy()
            else:
                x_adv = x_adv + alpha * np.sign(grads.numpy())
            
            x_adv = np.clip(x_adv, x - epsilon, x + epsilon)
            x_adv = np.clip(x_adv, 0, 1)
        
        return x_adv
    
    def evaluate_attack(self, X_test, y_test, attack_type='fgsm', epsilon=0.1, alpha=0.01, num_steps=10):
        """Evaluate model robustness against attack"""
        logger.info(f"Generating {attack_type.upper()} adversarial examples (epsilon={epsilon})")
        
        X_adv = np.zeros_like(X_test, dtype=np.float32)
        
        for i in tqdm(range(len(X_test)), desc=f"Generating {attack_type} examples"):
            x_single = X_test[i:i+1]
            y_single = y_test[i:i+1]
            
            if attack_type == 'fgsm':
                X_adv[i] = self.fgsm_attack(x_single, y_single, epsilon=epsilon).squeeze()
            elif attack_type == 'pgd':
                X_adv[i] = self.pgd_attack(x_single, y_single, epsilon=epsilon, alpha=alpha, num_steps=num_steps)
            elif attack_type == 'auto_pgd':
                X_adv[i] = self.auto_pgd_attack(x_single, y_single, epsilon=epsilon, num_steps=num_steps)
        
        # Evaluate on adversarial examples
        y_pred_adv = self.model(X_adv, training=False).numpy()
        y_pred_adv_binary = (y_pred_adv > 0.5).astype(np.float32)
        
        accuracy = accuracy_score(y_test, y_pred_adv_binary)
        precision = precision_score(y_test, y_pred_adv_binary, zero_division=0)
        recall = recall_score(y_test, y_pred_adv_binary, zero_division=0)
        f1 = f1_score(y_test, y_pred_adv_binary, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_adv)
        
        return X_adv, {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc)
        }


def compute_perturbation_magnitude(X_clean, X_adv):
    """Compute L-infinity perturbation magnitude"""
    diff = np.abs(X_clean - X_adv)
    linf = np.max(diff, axis=1)
    l2 = np.linalg.norm(diff, axis=1, ord=2)
    
    return {
        'linf_mean': float(np.mean(linf)),
        'linf_std': float(np.std(linf)),
        'l2_mean': float(np.mean(l2)),
        'l2_std': float(np.std(l2))
    }


def compute_mcnemar_test(y_true, y_pred_clean, y_pred_adv):
    """McNemar's test for paired comparison"""
    # Disagreement cases
    b = np.sum((y_pred_clean != y_true) & (y_pred_adv == y_true))
    c = np.sum((y_pred_clean == y_true) & (y_pred_adv != y_true))
    
    if b + c == 0:
        return {'statistic': 0, 'p_value': 1.0}
    
    statistic = ((b - c) ** 2) / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return {'statistic': float(statistic), 'p_value': float(p_value)}


def main(args):
    """Main evaluation pipeline"""
    
    logger.info("="*60)
    logger.info("FGSM & PGD Adversarial Robustness Evaluation")
    logger.info("="*60)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = keras.models.load_model(args.model_path)
    
    # Load test data
    logger.info(f"Loading test data from {args.data_path}")
    X_test = np.load(os.path.join(args.data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_path, 'y_test.npy'))
    
    logger.info(f"Test set: {X_test.shape}")
    
    # Get clean predictions
    logger.info("Computing clean predictions...")
    y_pred_clean = model(X_test, training=False).numpy()
    y_pred_clean_binary = (y_pred_clean > 0.5).astype(np.float32)
    clean_accuracy = accuracy_score(y_test, y_pred_clean_binary)
    logger.info(f"Clean Accuracy: {clean_accuracy:.4f}")
    
    # Adversarial evaluation
    evaluator = AdversarialEvaluator(model)
    results = {
        'clean': {
            'accuracy': float(clean_accuracy),
            'predictions': y_pred_clean_binary.tolist()
        }
    }
    
    # FGSM attack with different epsilons
    for epsilon in args.epsilon_list:
        logger.info(f"\nFGSM Attack (epsilon={epsilon})")
        X_fgsm, metrics = evaluator.evaluate_attack(
            X_test, y_test, attack_type='fgsm', epsilon=epsilon
        )
        results[f'fgsm_eps_{epsilon}'] = metrics
        
        # McNemar's test
        y_pred_fgsm = model(X_fgsm, training=False).numpy()
        y_pred_fgsm_binary = (y_pred_fgsm > 0.5).astype(np.float32)
        mcnemar = compute_mcnemar_test(y_test, y_pred_clean_binary, y_pred_fgsm_binary)
        results[f'fgsm_eps_{epsilon}']['mcnemar'] = mcnemar
        
        # Perturbation magnitude
        pert_mag = compute_perturbation_magnitude(X_test, X_fgsm)
        results[f'fgsm_eps_{epsilon}']['perturbation'] = pert_mag
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  McNemar p-value: {mcnemar['p_value']:.4f}")
        
        # Save examples (first 100)
        np.save(
            f'results/adversarial_examples/X_fgsm_eps_{epsilon}.npy',
            X_fgsm[:100]
        )
    
    # PGD-10 attack
    logger.info(f"\nPGD-10 Attack (epsilon={args.epsilon}, alpha={args.alpha})")
    X_pgd, metrics_pgd = evaluator.evaluate_attack(
        X_test, y_test, attack_type='pgd', epsilon=args.epsilon,
        alpha=args.alpha, num_steps=10
    )
    results['pgd_10'] = metrics_pgd
    
    y_pred_pgd = model(X_pgd, training=False).numpy()
    y_pred_pgd_binary = (y_pred_pgd > 0.5).astype(np.float32)
    mcnemar_pgd = compute_mcnemar_test(y_test, y_pred_clean_binary, y_pred_pgd_binary)
    results['pgd_10']['mcnemar'] = mcnemar_pgd
    
    pert_mag_pgd = compute_perturbation_magnitude(X_test, X_pgd)
    results['pgd_10']['perturbation'] = pert_mag_pgd
    
    logger.info(f"  Accuracy: {metrics_pgd['accuracy']:.4f}")
    logger.info(f"  McNemar p-value: {mcnemar_pgd['p_value']:.4f}")
    
    np.save('results/adversarial_examples/X_pgd_10.npy', X_pgd[:100])
    
    # Auto-PGD attack
    logger.info(f"\nAuto-PGD Attack (epsilon={args.epsilon})")
    X_autopgd, metrics_autopgd = evaluator.evaluate_attack(
        X_test, y_test, attack_type='auto_pgd', epsilon=args.epsilon
    )
    results['auto_pgd'] = metrics_autopgd
    
    y_pred_autopgd = model(X_autopgd, training=False).numpy()
    y_pred_autopgd_binary = (y_pred_autopgd > 0.5).astype(np.float32)
    mcnemar_autopgd = compute_mcnemar_test(y_test, y_pred_clean_binary, y_pred_autopgd_binary)
    results['auto_pgd']['mcnemar'] = mcnemar_autopgd
    
    pert_mag_autopgd = compute_perturbation_magnitude(X_test, X_autopgd)
    results['auto_pgd']['perturbation'] = pert_mag_autopgd
    
    logger.info(f"  Accuracy: {metrics_autopgd['accuracy']:.4f}")
    logger.info(f"  McNemar p-value: {mcnemar_autopgd['p_value']:.4f}")
    
    np.save('results/adversarial_examples/X_autopgd.npy', X_autopgd[:100])
    
    # Save comprehensive results
    with open('results/adversarial_robustness_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\nResults saved to results/adversarial_robustness_results.json")
    
    # Generate summary CSV
    summary_df = pd.DataFrame({
        'Attack': ['Clean', 'FGSM-0.05', 'FGSM-0.1', 'FGSM-0.2', 'PGD-10', 'Auto-PGD'],
        'Accuracy': [
            results['clean']['accuracy'],
            results['fgsm_eps_0.05']['accuracy'],
            results['fgsm_eps_0.1']['accuracy'],
            results['fgsm_eps_0.2']['accuracy'],
            results['pgd_10']['accuracy'],
            results['auto_pgd']['accuracy']
        ],
        'Precision': [
            'N/A',
            results['fgsm_eps_0.05']['precision'],
            results['fgsm_eps_0.1']['precision'],
            results['fgsm_eps_0.2']['precision'],
            results['pgd_10']['precision'],
            results['auto_pgd']['precision']
        ],
        'Recall': [
            'N/A',
            results['fgsm_eps_0.05']['recall'],
            results['fgsm_eps_0.1']['recall'],
            results['fgsm_eps_0.2']['recall'],
            results['pgd_10']['recall'],
            results['auto_pgd']['recall']
        ],
        'F1-Score': [
            'N/A',
            results['fgsm_eps_0.05']['f1'],
            results['fgsm_eps_0.1']['f1'],
            results['fgsm_eps_0.2']['f1'],
            results['pgd_10']['f1'],
            results['auto_pgd']['f1']
        ]
    })
    
    summary_df.to_csv('results/adversarial_summary.csv', index=False)
    logger.info("Summary saved to results/adversarial_summary.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial robustness evaluation')
    parser.add_argument('--model_path', type=str, default='models/xar_dnn_tf/xar_dnn.h5',
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, default='data/processed/',
                       help='Path to processed test data')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Perturbation budget')
    parser.add_argument('--alpha', type=float, default=0.01,
                       help='PGD step size')
    parser.add_argument('--epsilon_list', type=list, default=[0.05, 0.1, 0.2],
                       help='List of epsilons to test')
    
    args = parser.parse_args()
    main(args)
