"""
experiments/1_train_xar_dnn.py
Complete XAR-DNN Training Pipeline with Adversarial Training
Author: MD Hamid Borkot Tulla
"""

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directories
Path('logs').mkdir(exist_ok=True)
Path('models/xar_dnn_tf').mkdir(parents=True, exist_ok=True)
Path('results').mkdir(exist_ok=True)


class XARDNNModel(keras.Model):
    """
    XAR-DNN: eXplainable Adversarially Robust Deep Neural Network
    Architecture: 42 -> 128 -> 64 -> 32 -> 1
    """
    
    def __init__(self, input_dim=42, dropout_rate=0.3, **kwargs):
        super(XARDNNModel, self).__init__(**kwargs)
        
        self.dense1 = layers.Dense(128, activation=None, kernel_regularizer=regularizers.l2(1e-5))
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.relu1 = layers.ReLU()
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.dense2 = layers.Dense(64, activation=None, kernel_regularizer=regularizers.l2(1e-5))
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.relu2 = layers.ReLU()
        self.dropout2 = layers.Dropout(dropout_rate)
        
        self.dense3 = layers.Dense(32, activation=None, kernel_regularizer=regularizers.l2(1e-5))
        self.ln3 = layers.LayerNormalization(epsilon=1e-6)
        self.relu3 = layers.ReLU()
        self.dropout3 = layers.Dropout(dropout_rate)
        
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.dropout2(x, training=training)
        
        x = self.dense3(x)
        x = self.ln3(x)
        x = self.relu3(x)
        x = self.dropout3(x, training=training)
        
        return self.output_layer(x)


class AdversarialTrainer:
    """
    Adversarial training with PGD attacks
    """
    
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = keras.losses.BinaryCrossentropy()
        
    @tf.function
    def fgsm_attack(self, x, y, epsilon=0.1):
        """Fast Gradient Sign Method"""
        x_tensor = tf.Variable(x, trainable=True)
        
        with tf.GradientTape() as tape:
            logits = self.model(x_tensor, training=True)
            loss = self.loss_fn(y, logits)
        
        grads = tape.gradient(loss, x_tensor)
        x_adv = x + epsilon * tf.sign(grads)
        x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)
        
        return x_adv
    
    @tf.function
    def pgd_attack(self, x, y, epsilon=0.1, alpha=0.01, num_steps=10):
        """Projected Gradient Descent Attack"""
        x_adv = x + tf.random.uniform(tf.shape(x), -epsilon, epsilon)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
        
        for _ in tf.range(num_steps):
            x_tensor = tf.Variable(x_adv, trainable=True)
            
            with tf.GradientTape() as tape:
                logits = self.model(x_tensor, training=True)
                loss = self.loss_fn(y, logits)
            
            grads = tape.gradient(loss, x_tensor)
            x_adv = x_adv + alpha * tf.sign(grads)
            x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)
            x_adv = tf.clip_by_value(x_adv, 0, 1)
        
        return x_adv
    
    def train_step(self, x_batch, y_batch, epsilon=0.1, alpha=0.01, num_pgd_steps=10):
        """Single training step with clean + adversarial loss"""
        
        # Clean training
        with tf.GradientTape() as tape:
            logits_clean = self.model(x_batch, training=True)
            loss_clean = self.loss_fn(y_batch, logits_clean)
        
        grads_clean = tape.gradient(loss_clean, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads_clean, self.model.trainable_weights))
        
        # Adversarial training (PGD)
        x_adv = self.pgd_attack(x_batch, y_batch, epsilon=epsilon, alpha=alpha, num_steps=num_pgd_steps)
        
        with tf.GradientTape() as tape:
            logits_adv = self.model(x_adv, training=True)
            loss_adv = self.loss_fn(y_batch, logits_adv)
        
        grads_adv = tape.gradient(loss_adv, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads_adv, self.model.trainable_weights))
        
        return float(loss_clean), float(loss_adv)


def load_edge_iiotset(data_path='data/Edge-IIoTSet/', split_ratio=0.8, seed=42):
    """Load and preprocess Edge-IIoTSet dataset"""
    logger.info(f"Loading Edge-IIoTSet from {data_path}")
    
    # Load CSV
    csv_file = os.path.join(data_path, 'Edge-IIoTSet.csv')
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset not found at {csv_file}")
    
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Select 42 numeric features (exclude ID columns and labels)
    feature_cols = [col for col in df.columns if col not in ['Label', 'Attack', 'Flow_ID', 'Src_IP', 'Dst_IP']]
    feature_cols = feature_cols[:42]  # Select top 42 features
    
    X = df[feature_cols].values.astype(np.float32)
    y = (df['Label'] != 'Normal').astype(np.float32).values  # Binary: Normal(0) vs Attack(1)
    
    logger.info(f"Class distribution - Normal: {np.sum(y==0)}, Attack: {np.sum(y==1)}")
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Normalize features (Z-score)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-split_ratio, random_state=seed, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Save scaler
    joblib.dump(scaler, 'models/xar_dnn_tf/scaler.pkl')
    
    return X_train, X_test, y_train, y_test, scaler, feature_cols


def train_model(X_train, y_train, X_val, y_val, 
                batch_size=512, epochs=60, epsilon=0.1, alpha=0.01, num_pgd_steps=10):
    """Train XAR-DNN with adversarial training"""
    
    logger.info("Initializing XAR-DNN model...")
    model = XARDNNModel(input_dim=42, dropout_rate=0.3)
    trainer = AdversarialTrainer(model, learning_rate=1e-3)
    
    # Build model
    model.build(input_shape=(None, 42))
    logger.info(f"Model parameters: {model.count_params():,}")
    
    history = {
        'epoch': [], 'loss_clean': [], 'loss_adv': [],
        'val_accuracy': [], 'val_auc': []
    }
    
    num_batches = len(X_train) // batch_size
    best_val_accuracy = 0
    patience_counter = 0
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss_clean = []
        epoch_loss_adv = []
        
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Training batches
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(X_train))
            
            x_batch = tf.convert_to_tensor(X_train_shuffled[start_idx:end_idx])
            y_batch = tf.convert_to_tensor(y_train_shuffled[start_idx:end_idx].reshape(-1, 1))
            
            loss_clean, loss_adv = trainer.train_step(
                x_batch, y_batch, epsilon=epsilon, alpha=alpha, num_pgd_steps=num_pgd_steps
            )
            
            epoch_loss_clean.append(loss_clean)
            epoch_loss_adv.append(loss_adv)
            
            pbar.set_postfix({
                'loss_clean': np.mean(epoch_loss_clean),
                'loss_adv': np.mean(epoch_loss_adv)
            })
        
        # Validation
        y_val_pred = model(X_val, training=False).numpy()
        y_val_pred_binary = (y_val_pred > 0.5).astype(np.float32)
        
        val_accuracy = accuracy_score(y_val, y_val_pred_binary)
        val_auc = roc_auc_score(y_val, y_val_pred)
        
        history['epoch'].append(epoch + 1)
        history['loss_clean'].append(np.mean(epoch_loss_clean))
        history['loss_adv'].append(np.mean(epoch_loss_adv))
        history['val_accuracy'].append(val_accuracy)
        history['val_auc'].append(val_auc)
        
        logger.info(
            f"Epoch {epoch+1}: loss_clean={np.mean(epoch_loss_clean):.4f}, "
            f"loss_adv={np.mean(epoch_loss_adv):.4f}, "
            f"val_accuracy={val_accuracy:.4f}, val_auc={val_auc:.4f}"
        )
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            model.save_weights('models/xar_dnn_tf/xar_dnn_best.h5')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save final model
    model.save('models/xar_dnn_tf/xar_dnn.h5')
    logger.info("Model saved to models/xar_dnn_tf/xar_dnn.h5")
    
    return model, history


def evaluate_model(model, X_test, y_test):
    """Evaluate model on clean test set"""
    logger.info("Evaluating model on clean test set...")
    
    y_pred = model(X_test, training=False).numpy()
    y_pred_binary = (y_pred > 0.5).astype(np.float32)
    
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    auc = roc_auc_score(y_test, y_pred)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc)
    }
    
    logger.info(f"Clean Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"AUC-ROC: {auc:.4f}")
    
    return results


def main(args):
    """Main training pipeline"""
    
    logger.info("="*60)
    logger.info("XAR-DNN Training Pipeline")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test, scaler, feature_cols = load_edge_iiotset(
        data_path=args.data_path,
        split_ratio=0.8,
        seed=42
    )
    
    # Split train into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    logger.info(f"Final splits - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Train model
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_pgd_steps=args.pgd_steps
    )
    
    # Evaluate
    test_results = evaluate_model(model, X_test, y_test)
    
    # Save results
    with open('results/training_results.json', 'w') as f:
        json.dump({
            'test_metrics': test_results,
            'training_history': history,
            'config': {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': 1e-3,
                'epsilon': args.epsilon,
                'alpha': args.alpha,
                'pgd_steps': args.pgd_steps
            }
        }, f, indent=2)
    
    logger.info("Training complete. Results saved to results/training_results.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train XAR-DNN model')
    parser.add_argument('--data_path', type=str, default='data/Edge-IIoTSet/', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Perturbation epsilon')
    parser.add_argument('--alpha', type=float, default=0.01, help='PGD step size')
    parser.add_argument('--pgd_steps', type=int, default=10, help='PGD attack steps')
    
    args = parser.parse_args()
    main(args)
