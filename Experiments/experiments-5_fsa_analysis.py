"""
experiments/5_fsa_analysis.py
Feature Stability Analysis (FSA) - Core FSA Computation
Author: MD Hamid Borkot Tulla
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
import shap
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fsa_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

Path('results').mkdir(exist_ok=True)
Path('results/fsa_analysis').mkdir(exist_ok=True)


class FeatureStabilityAnalyzer:
    """
    Feature Stability Analysis (FSA)
    Quantifies SHAP attribution consistency under adversarial perturbation
    
    FSA metric: S_i = 1 - ||φ_i^adv - φ_i^clean||_2 / (||φ_i^clean||_2 + ε)
    where:
      - φ_i^clean: SHAP value for feature i on clean input
      - φ_i^adv: SHAP value for feature i on adversarial input
      - ε = 1e-6 (numerical stability)
    """
    
    def __init__(self, model, background_data=None, num_background=200):
        """
        Args:
            model: Keras/TF model
            background_data: Background samples for SHAP
            num_background: Number of background samples
        """
        self.model = model
        self.num_background = num_background
        
        if background_data is not None:
            self.background_data = background_data[:num_background]
        else:
            self.background_data = None
    
    def compute_shap_values(self, X, background_data=None):
        """
        Compute SHAP values using KernelSHAP
        
        Args:
            X: Input samples
            background_data: Background samples
        
        Returns:
            SHAP values (num_samples, num_features)
        """
        if background_data is None:
            background_data = self.background_data
        
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        
        # Create explainer
        explainer = shap.KernelExplainer(
            model=lambda x: self.model(x, training=False).numpy().squeeze(),
            data=background_data,
            link='logit'
        )
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X)
        
        return shap_values
    
    def compute_fsa_scores(self, X_clean, X_adv, background_data=None, epsilon=1e-6):
        """
        Compute Feature Stability Analysis scores
        
        Args:
            X_clean: Clean input samples
            X_adv: Adversarial input samples
            background_data: Background samples for SHAP
            epsilon: Numerical stability constant
        
        Returns:
            Dictionary with FSA results
        """
        logger.info("Computing Feature Stability Analysis...")
        
        # Compute SHAP values
        logger.info("Step 1: Computing SHAP for clean samples...")
        shap_clean = self.compute_shap_values(X_clean, background_data)
        
        logger.info("Step 2: Computing SHAP for adversarial samples...")
        shap_adv = self.compute_shap_values(X_adv, background_data)
        
        # Handle multi-output case (if needed)
        if len(shap_clean.shape) > 2:
            shap_clean = shap_clean[:, :, 1]  # Take attack class
            shap_adv = shap_adv[:, :, 1]
        
        # Compute stability scores per feature
        logger.info("Step 3: Computing stability scores...")
        num_samples, num_features = shap_clean.shape
        stability_scores = np.zeros((num_samples, num_features))
        
        for i in range(num_samples):
            for j in range(num_features):
                numerator = np.abs(shap_adv[i, j] - shap_clean[i, j])
                denominator = np.abs(shap_clean[i, j]) + epsilon
                
                # S_i ∈ [0, 1]
                s_ij = 1 - (numerator / denominator)
                stability_scores[i, j] = np.clip(s_ij, 0, 1)
        
        # Aggregate across samples
        mean_stability = np.mean(stability_scores, axis=0)
        std_stability = np.std(stability_scores, axis=0)
        
        # Count high-stability features (S_i > 0.7)
        high_stability_mask = mean_stability > 0.7
        num_high_stability = np.sum(high_stability_mask)
        pct_high_stability = 100 * num_high_stability / num_features
        
        logger.info(f"Mean FSA: {np.mean(mean_stability):.4f}")
        logger.info(f"High-stability features (S_i > 0.7): {num_high_stability}/{num_features} ({pct_high_stability:.1f}%)")
        
        return {
            'stability_scores': stability_scores,
            'mean_stability': mean_stability,
            'std_stability': std_stability,
            'high_stability_mask': high_stability_mask,
            'num_high_stability': int(num_high_stability),
            'pct_high_stability': float(pct_high_stability),
            'shap_clean': shap_clean,
            'shap_adv': shap_adv
        }
    
    def compute_explanation_subversion_rate(self, stability_scores, top_k=3):
        """
        Compute Explanation Subversion Rate (ESR)
        Percentage of samples where top-k stable features become unstable
        
        Args:
            stability_scores: FSA scores (num_samples, num_features)
            top_k: Number of top features to track
        
        Returns:
            ESR percentage
        """
        num_samples = stability_scores.shape[0]
        subversions = 0
        
        for i in range(num_samples):
            # Get top-k most stable features (highest S_i in clean)
            top_indices = np.argsort(stability_scores[i])[-top_k:]
            
            # Check if they become unstable (S_i < 0.5)
            if np.any(stability_scores[i, top_indices] < 0.5):
                subversions += 1
        
        esr = 100 * subversions / num_samples
        logger.info(f"Explanation Subversion Rate (ESR): {esr:.1f}%")
        
        return esr


def evaluate_fsa_across_attacks(model, X_clean, X_adv_dict, background_data, feature_names):
    """
    Evaluate FSA across different adversarial attacks
    
    Args:
        model: Trained model
        X_clean: Clean samples
        X_adv_dict: Dictionary of adversarial samples by attack type
        background_data: Background samples
        feature_names: Feature names
    
    Returns:
        FSA results dictionary
    """
    analyzer = FeatureStabilityAnalyzer(model, background_data)
    results = {}
    
    for attack_type, X_adv in X_adv_dict.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"FSA Analysis: {attack_type.upper()}")
        logger.info(f"{'='*60}")
        
        fsa_results = analyzer.compute_fsa_scores(X_clean, X_adv, background_data)
        
        # Compute ESR
        esr = analyzer.compute_explanation_subversion_rate(fsa_results['stability_scores'])
        fsa_results['esr'] = float(esr)
        
        # Rank features by stability
        mean_stability = fsa_results['mean_stability']
        feature_ranking = np.argsort(mean_stability)[::-1]
        
        top_features = []
        for rank, feat_idx in enumerate(feature_ranking[:10]):
            top_features.append({
                'rank': int(rank + 1),
                'feature_name': feature_names[feat_idx],
                'stability_score': float(mean_stability[feat_idx]),
                'std': float(fsa_results['std_stability'][feat_idx])
            })
        
        fsa_results['top_10_stable_features'] = top_features
        
        # Remove large arrays from serialization
        fsa_results.pop('stability_scores', None)
        fsa_results.pop('shap_clean', None)
        fsa_results.pop('shap_adv', None)
        fsa_results.pop('high_stability_mask', None)
        
        results[attack_type] = fsa_results
    
    return results


def main(args):
    """Main FSA analysis pipeline"""
    
    logger.info("="*60)
    logger.info("Feature Stability Analysis (FSA) Computation")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*60)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = keras.models.load_model(args.model_path)
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    X_clean = np.load(os.path.join(args.data_path, 'X_val.npy'))
    background_data = np.load(os.path.join(args.data_path, 'X_train.npy'))[:200]
    
    logger.info(f"Clean samples: {X_clean.shape}")
    logger.info(f"Background samples: {background_data.shape}")
    
    # Load adversarial examples (or generate on-the-fly)
    X_adv_dict = {}
    
    adversarial_files = [
        ('fgsm_0.1', os.path.join(args.data_path, 'X_fgsm_0.1.npy')),
        ('pgd_10', os.path.join(args.data_path, 'X_pgd_10.npy')),
        ('auto_pgd', os.path.join(args.data_path, 'X_autopgd.npy'))
    ]
    
    for attack_name, file_path in adversarial_files:
        if os.path.exists(file_path):
            X_adv_dict[attack_name] = np.load(file_path)
            logger.info(f"Loaded {attack_name}: {X_adv_dict[attack_name].shape}")
    
    # Feature names
    feature_names = [
        'flow_duration', 'packet_rate', 'tcp.flags', 'payload_length',
        'packet_count', 'protocol_type', 'src_port', 'dst_port',
        'flow_iat_mean', 'flow_iat_std', 'packet_iat_mean', 'packet_iat_std',
        'tcp.len', 'tcp.ack', 'tcp.syn', 'tcp.fin',
        'udp.length', 'icmp.checksum', 'dns.qry.type', 'dns.qry.name.len',
        'mqtt.msgtype', 'mqtt.qos', 'http.method', 'http.uri.len',
        'ssl.version', 'ssl.record.length', 'ftp.command', 'ftp.arg.len',
        'modbus.func', 'modbus.data', 'dnp3.func', 'dnp3.data',
        'dhcp.type', 'dhcp.secs', 'ntp.version', 'ntp.stratum',
        'snmp.version', 'snmp.pdu.type', 'kerberos.msgtype', 'ldap.msgtype',
        'radius.code', 'tacacs.type'
    ]
    
    # Evaluate FSA
    results = evaluate_fsa_across_attacks(model, X_clean, X_adv_dict, background_data, feature_names)
    
    # Save results
    with open('results/fsa_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\nFSA results saved to results/fsa_analysis_results.json")
    
    # Generate summary table
    summary_data = []
    for attack_type, fsa_data in results.items():
        summary_data.append({
            'Attack': attack_type,
            'Mean FSA': fsa_data['mean_stability'].mean() if isinstance(fsa_data['mean_stability'], np.ndarray) else fsa_data['mean_stability'],
            'High-Stability %': fsa_data['pct_high_stability'],
            'ESR %': fsa_data['esr']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/fsa_summary.csv', index=False)
    logger.info("FSA summary saved to results/fsa_summary.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Stability Analysis')
    parser.add_argument('--model_path', type=str, default='models/xar_dnn_tf/xar_dnn.h5',
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, default='data/processed/',
                       help='Path to processed data')
    parser.add_argument('--num_background', type=int, default=200,
                       help='Number of background samples for SHAP')
    
    args = parser.parse_args()
    main(args)
