"""
experiments/4_semantic_attacks.py
Semantic/Protocol-Level Adversarial Attacks on IoT Traffic
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/semantic_attacks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

Path('results').mkdir(exist_ok=True)
Path('results/semantic_attacks').mkdir(exist_ok=True)


class SemanticAttacker:
    """
    Generate semantically meaningful adversarial examples targeting IoT protocols
    """
    
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names or [f'feature_{i}' for i in range(42)]
    
    def mqtt_replay_attack(self, X, rate=0.1):
        """
        MQTT Protocol Attack: Replay valid MQTT messages
        Simulates replaying captured MQTT packets
        
        Args:
            X: Input samples
            rate: Replay rate (fraction of flows to duplicate)
        
        Returns:
            Modified samples
        """
        X_adv = X.copy()
        num_replay = int(len(X) * rate)
        
        replay_indices = np.random.choice(len(X), num_replay, replace=False)
        
        for idx in replay_indices:
            # Replay increases flow duration and packet rate
            X_adv[idx, 0] *= 1.5  # flow_duration
            X_adv[idx, 1] *= 1.3  # packet_rate
        
        return X_adv
    
    def packet_drop_attack(self, X, drop_rate=0.1):
        """
        Packet Loss Attack: Simulate packet drops in IoT flows
        
        Args:
            X: Input samples
            drop_rate: Fraction of packets to drop
        
        Returns:
            Modified samples
        """
        X_adv = X.copy()
        
        # Packet drops reduce flow metrics
        drop_mask = np.random.random((len(X), X.shape[1])) < drop_rate
        
        for i in range(len(X)):
            if np.any(drop_mask[i]):
                X_adv[i, drop_mask[i]] *= 0.7  # Reduce affected features
        
        return X_adv
    
    def packet_jitter_attack(self, X, jitter_percent=0.05, max_ms=200):
        """
        Timing Attack: Add jitter to inter-packet timing
        
        Args:
            X: Input samples
            jitter_percent: Percentage jitter
            max_ms: Maximum time warp in milliseconds
        
        Returns:
            Modified samples
        """
        X_adv = X.copy()
        
        # Apply timing perturbations
        jitter_noise = np.random.uniform(-max_ms, max_ms, X_adv.shape)
        timing_mask = np.abs(jitter_noise) > 0
        
        # Affect timing-related features
        X_adv = X_adv + jitter_noise * jitter_percent
        X_adv = np.clip(X_adv, 0, 1)
        
        return X_adv
    
    def protocol_violation_attack(self, X, violation_rate=0.05):
        """
        Protocol Anomaly: Violate protocol-specific constraints
        
        Args:
            X: Input samples
            violation_rate: Fraction of flows with violations
        
        Returns:
            Modified samples
        """
        X_adv = X.copy()
        
        num_violations = int(len(X) * violation_rate)
        violation_indices = np.random.choice(len(X), num_violations, replace=False)
        
        for idx in violation_indices:
            # Violate protocol constraints
            # tcp.flags inconsistency
            if len(X_adv[idx]) > 5:
                X_adv[idx, 5] = np.random.random()  # tcp.flags anomaly
            
            # Protocol mismatch
            if len(X_adv[idx]) > 10:
                X_adv[idx, 10] = np.random.random()  # Protocol field
        
        return X_adv
    
    def combined_semantic_attack(self, X, attack_types=['mqtt_replay', 'packet_jitter']):
        """Combine multiple semantic attacks"""
        X_adv = X.copy()
        
        if 'mqtt_replay' in attack_types:
            X_adv = self.mqtt_replay_attack(X_adv, rate=0.05)
        
        if 'packet_drop' in attack_types:
            X_adv = self.packet_drop_attack(X_adv, drop_rate=0.05)
        
        if 'packet_jitter' in attack_types:
            X_adv = self.packet_jitter_attack(X_adv, jitter_percent=0.05)
        
        if 'protocol_violation' in attack_types:
            X_adv = self.protocol_violation_attack(X_adv, violation_rate=0.05)
        
        return X_adv


def compute_feature_stability_analysis(model, X_clean, X_adv, y_true, feature_names):
    """
    Compute Feature Stability Analysis (FSA)
    Measure SHAP consistency between clean and adversarial inputs
    """
    logger.info("Computing Feature Stability Analysis (FSA)...")
    
    # Get SHAP values for clean samples
    logger.info("Computing SHAP values for clean samples...")
    explainer = shap.KernelExplainer(
        lambda x: model(x, training=False).numpy(),
        data=shap.sample(X_clean, 200)
    )
    
    shap_clean = explainer.shap_values(X_clean[:100])  # Subset for efficiency
    shap_adv = explainer.shap_values(X_adv[:100])
    
    # Compute stability scores
    stability_scores = []
    epsilon = 1e-6
    
    for i in range(len(shap_clean)):
        numerator = np.linalg.norm(shap_adv[i] - shap_clean[i], ord=2)
        denominator = np.linalg.norm(shap_clean[i], ord=2) + epsilon
        
        s_i = 1 - (numerator / denominator)
        stability_scores.append(max(0, min(1, s_i)))
    
    fsa_mean = np.mean(stability_scores)
    
    logger.info(f"FSA (Feature Stability): {fsa_mean:.4f}")
    
    return {
        'fsa_scores': stability_scores,
        'fsa_mean': float(fsa_mean),
        'fsa_std': float(np.std(stability_scores))
    }


def evaluate_semantic_attacks(model, X_test, y_test, feature_names):
    """Evaluate model robustness to semantic attacks"""
    
    logger.info("="*60)
    logger.info("Evaluating Semantic Attacks")
    logger.info("="*60)
    
    attacker = SemanticAttacker(model, feature_names)
    
    results = {
        'clean': {},
        'mqtt_replay': {},
        'packet_drop': {},
        'packet_jitter': {},
        'protocol_violation': {},
        'combined': {}
    }
    
    # Clean baseline
    logger.info("\nClean baseline:")
    y_pred_clean = model(X_test, training=False).numpy()
    y_pred_clean_binary = (y_pred_clean > 0.5).astype(np.float32)
    
    clean_accuracy = accuracy_score(y_test, y_pred_clean_binary)
    clean_precision = precision_score(y_test, y_pred_clean_binary, zero_division=0)
    clean_recall = recall_score(y_test, y_pred_clean_binary, zero_division=0)
    clean_f1 = f1_score(y_test, y_pred_clean_binary, zero_division=0)
    
    results['clean'] = {
        'accuracy': float(clean_accuracy),
        'precision': float(clean_precision),
        'recall': float(clean_recall),
        'f1': float(clean_f1)
    }
    logger.info(f"Clean Accuracy: {clean_accuracy:.4f}")
    
    # MQTT Replay Attack
    logger.info("\nMQTT Replay Attack:")
    X_mqtt = attacker.mqtt_replay_attack(X_test, rate=0.1)
    y_pred_mqtt = model(X_mqtt, training=False).numpy()
    y_pred_mqtt_binary = (y_pred_mqtt > 0.5).astype(np.float32)
    
    mqtt_accuracy = accuracy_score(y_test, y_pred_mqtt_binary)
    results['mqtt_replay'] = {
        'accuracy': float(mqtt_accuracy),
        'robustness_delta': float(clean_accuracy - mqtt_accuracy)
    }
    logger.info(f"MQTT Replay Accuracy: {mqtt_accuracy:.4f} (Δ={clean_accuracy - mqtt_accuracy:.4f})")
    
    fsa_mqtt = compute_feature_stability_analysis(model, X_test, X_mqtt, y_test, feature_names)
    results['mqtt_replay']['fsa'] = fsa_mqtt
    
    # Packet Drop Attack
    logger.info("\nPacket Drop Attack:")
    X_drop = attacker.packet_drop_attack(X_test, drop_rate=0.1)
    y_pred_drop = model(X_drop, training=False).numpy()
    y_pred_drop_binary = (y_pred_drop > 0.5).astype(np.float32)
    
    drop_accuracy = accuracy_score(y_test, y_pred_drop_binary)
    results['packet_drop'] = {
        'accuracy': float(drop_accuracy),
        'robustness_delta': float(clean_accuracy - drop_accuracy)
    }
    logger.info(f"Packet Drop Accuracy: {drop_accuracy:.4f} (Δ={clean_accuracy - drop_accuracy:.4f})")
    
    # Packet Jitter Attack
    logger.info("\nPacket Jitter Attack:")
    X_jitter = attacker.packet_jitter_attack(X_test, jitter_percent=0.05)
    y_pred_jitter = model(X_jitter, training=False).numpy()
    y_pred_jitter_binary = (y_pred_jitter > 0.5).astype(np.float32)
    
    jitter_accuracy = accuracy_score(y_test, y_pred_jitter_binary)
    results['packet_jitter'] = {
        'accuracy': float(jitter_accuracy),
        'robustness_delta': float(clean_accuracy - jitter_accuracy)
    }
    logger.info(f"Packet Jitter Accuracy: {jitter_accuracy:.4f} (Δ={clean_accuracy - jitter_accuracy:.4f})")
    
    # Protocol Violation Attack
    logger.info("\nProtocol Violation Attack:")
    X_protocol = attacker.protocol_violation_attack(X_test, violation_rate=0.1)
    y_pred_protocol = model(X_protocol, training=False).numpy()
    y_pred_protocol_binary = (y_pred_protocol > 0.5).astype(np.float32)
    
    protocol_accuracy = accuracy_score(y_test, y_pred_protocol_binary)
    results['protocol_violation'] = {
        'accuracy': float(protocol_accuracy),
        'robustness_delta': float(clean_accuracy - protocol_accuracy)
    }
    logger.info(f"Protocol Violation Accuracy: {protocol_accuracy:.4f} (Δ={clean_accuracy - protocol_accuracy:.4f})")
    
    # Combined Attack
    logger.info("\nCombined Semantic Attack:")
    X_combined = attacker.combined_semantic_attack(X_test, attack_types=['mqtt_replay', 'packet_jitter'])
    y_pred_combined = model(X_combined, training=False).numpy()
    y_pred_combined_binary = (y_pred_combined > 0.5).astype(np.float32)
    
    combined_accuracy = accuracy_score(y_test, y_pred_combined_binary)
    results['combined'] = {
        'accuracy': float(combined_accuracy),
        'robustness_delta': float(clean_accuracy - combined_accuracy)
    }
    logger.info(f"Combined Attack Accuracy: {combined_accuracy:.4f} (Δ={clean_accuracy - combined_accuracy:.4f})")
    
    return results


def main(args):
    """Main evaluation pipeline"""
    
    logger.info("="*60)
    logger.info("Semantic Attacks Evaluation")
    logger.info("="*60)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = keras.models.load_model(args.model_path)
    
    # Load test data
    logger.info(f"Loading test data from {args.data_path}")
    X_test = np.load(os.path.join(args.data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_path, 'y_test.npy'))
    
    logger.info(f"Test set: {X_test.shape}")
    
    # Feature names (42 features from Edge-IIoTSet)
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
    
    # Evaluate semantic attacks
    results = evaluate_semantic_attacks(model, X_test, y_test, feature_names)
    
    # Save results
    with open('results/semantic_attacks_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\nResults saved to results/semantic_attacks_results.json")
    
    # Generate summary CSV
    summary_df = pd.DataFrame({
        'Attack Type': list(results.keys()),
        'Accuracy': [results[k].get('accuracy', 'N/A') for k in results.keys()],
        'Robustness Delta': [results[k].get('robustness_delta', 'N/A') for k in results.keys()]
    })
    
    summary_df.to_csv('results/semantic_attacks_summary.csv', index=False)
    logger.info("Summary saved to results/semantic_attacks_summary.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic attacks evaluation')
    parser.add_argument('--model_path', type=str, default='models/xar_dnn_tf/xar_dnn.h5',
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, default='data/processed/',
                       help='Path to processed test data')
    
    args = parser.parse_args()
    main(args)
