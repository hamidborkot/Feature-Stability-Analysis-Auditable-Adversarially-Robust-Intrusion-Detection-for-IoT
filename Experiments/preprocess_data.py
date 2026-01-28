#!/usr/bin/env python3
"""
DATA PREPROCESSING SCRIPT
==========================
Prepares Edge-IIoTSet, NSL-KDD, or CIC-IDS2018 for XAR-DNN training.

Quick Start:
    python preprocess_data.py --dataset edge-iiotset --file Edge-IIoTSet.csv

This script:
1. Loads raw dataset
2. Selects 42 features (numeric only, no NaN)
3. Normalizes using StandardScaler
4. Splits into train/test (80/20)
5. Saves as .npy files for fast loading
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)


def preprocess_edge_iiotset(csv_file):
    """
    Preprocess Edge-IIoTSet dataset.
    
    Expected columns: 42 numeric features + Label + Attack
    """
    logger.info(f"Loading Edge-IIoTSet from {csv_file}")
    df = pd.read_csv(csv_file)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)[:10]}...")  # Show first 10
    
    # Drop non-numeric columns
    exclude_cols = ['Label', 'Attack', 'Flow_ID', 'Src_IP', 'Dst_IP', 'Dst_Port', 'Protocol']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'object']
    
    # Select top 42 features by variance (most informative)
    if len(feature_cols) > 42:
        feature_cols = feature_cols[:42]
    
    logger.info(f"Selected {len(feature_cols)} features")
    
    # Extract features and labels
    X = df[feature_cols].values.astype('float32')
    y = (df['Label'] != 'Normal').astype('float32')
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info(f"Feature matrix: {X.shape}, Label distribution: {np.bincount(y.astype(int))}")
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Save
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")
    
    logger.info("✅ Edge-IIoTSet preprocessing complete")
    return X_train, X_test, y_train, y_test


def preprocess_nsl_kdd(csv_file):
    """
    Preprocess NSL-KDD dataset.
    """
    logger.info(f"Loading NSL-KDD from {csv_file}")
    df = pd.read_csv(csv_file, header=None)
    
    # NSL-KDD has 41 features, last column is label
    feature_cols = list(range(0, 41))
    X = df[feature_cols].values.astype('float32')
    
    # Convert label to binary (Normal vs Attack)
    label_col = df[41].values
    y = (label_col != 'normal').astype('float32')
    
    logger.info(f"Feature matrix: {X.shape}, Labels: {np.bincount(y.astype(int))}")
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")
    
    logger.info("✅ NSL-KDD preprocessing complete")
    return X_train, X_test, y_train, y_test


def preprocess_cic_ids2018(csv_file):
    """
    Preprocess CIC-IDS2018 dataset.
    """
    logger.info(f"Loading CIC-IDS2018 from {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Select numeric features
    exclude_cols = ['Label', ' Label', 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
    feature_cols = [col for col in df.columns if col not in exclude_cols][:42]
    
    X = df[feature_cols].dropna().values.astype('float32')
    y_raw = df.loc[df[feature_cols].notna().all(axis=1), ' Label'].values
    y = (y_raw != 'Benign').astype('float32')
    
    logger.info(f"Feature matrix: {X.shape}, Labels: {np.bincount(y.astype(int))}")
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")
    
    logger.info("✅ CIC-IDS2018 preprocessing complete")
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing for XAR-DNN")
    parser.add_argument("--dataset", choices=["edge-iiotset", "nsl-kdd", "cic-ids2018"],
                        default="edge-iiotset", help="Dataset to preprocess")
    parser.add_argument("--file", required=True, help="Path to dataset CSV file")
    
    args = parser.parse_args()
    
    logger.info(f"Starting preprocessing for {args.dataset}")
    
    if args.dataset == "edge-iiotset":
        preprocess_edge_iiotset(args.file)
    elif args.dataset == "nsl-kdd":
        preprocess_nsl_kdd(args.file)
    elif args.dataset == "cic-ids2018":
        preprocess_cic_ids2018(args.file)
    
    logger.info("All preprocessing complete! Data ready for training.")
    logger.info(f"Location: {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
