"""
Model Training Script
Trains multiple ML models (XGBoost, Random Forest, LSTM, Autoencoder) 
for anomaly detection in EV charging data
"""

import pandas as pd
import numpy as np
import logging
import yaml
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates ML models for anomaly detection"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config["model_training"]
        self.target_tpr = self.model_config.get("target_tpr", 0.95)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        
    def load_features(self, feature_file: str) -> pd.DataFrame:
        """Load feature dataset"""
        logger.info(f"Loading features from {feature_file}...")
        
        if feature_file.endswith(".parquet"):
            df = pd.read_parquet(feature_file)
        else:
            df = pd.read_csv(feature_file)
        
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame):
        """Prepare data for training"""
        logger.info("Preparing data...")
        
        # Separate features and labels
        if "anomaly_label" not in df.columns:
            raise ValueError("Dataset must contain 'anomaly_label' column")
        
        # Drop non-feature columns
        drop_cols = ["timestamp", "charge_point_id", "anomaly_label"]
        feature_cols = [col for col in df.columns if col not in drop_cols]
        
        X = df[feature_cols].copy()
        y = df["anomaly_label"].copy()
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            X[col] = X[col].astype("category").cat.codes
        
        # Handle missing values
        X.fillna(0, inplace=True)
        
        # Handle infinite values
        X.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"Features: {X.shape[1]}")
        logger.info(f"Samples: {len(X)}")
        logger.info(f"Classes: {len(self.label_encoder.classes_)}")
        logger.info(f"Class names: {list(self.label_encoder.classes_)}")
        
        # Print class distribution
        unique, counts = np.unique(y_encoded, return_counts=True)
        logger.info("\nClass distribution:")
        for cls_idx, count in zip(unique, counts):
            cls_name = self.label_encoder.classes_[cls_idx]
            pct = (count / len(y_encoded)) * 100
            logger.info(f"  {cls_name:25s}: {count:6d} ({pct:5.1f}%)")
        
        return X, y_encoded, feature_cols
    
    def split_data(self, X, y):
        """Split data into train/val/test sets"""
        train_ratio = self.model_config["train_split"]
        val_ratio = self.model_config["val_split"]
        test_ratio = self.model_config["test_split"]
        
        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        val_adjusted_ratio = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_adjusted_ratio, 
            random_state=42, stratify=y_temp
        )
        
        logger.info(f"\nData split:")
        logger.info(f"  Train: {len(X_train):6d} samples ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"  Val:   {len(X_val):6d} samples ({len(X_val)/len(X)*100:.1f}%)")
        logger.info(f"  Test:  {len(X_test):6d} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def balance_data(self, X_train, y_train):
        """Balance training data using SMOTE"""
        if not self.model_config.get("balance_classes", True):
            return X_train, y_train
        
        logger.info("\nBalancing training data with SMOTE...")
        
        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        min_samples = np.min(counts)
        
        if min_samples < 6:
            logger.warning(f"Minimum samples ({min_samples}) < 6, skipping SMOTE")
            return X_train, y_train
        
        try:
            # Define resampling strategy
            smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples-1))
            under = RandomUnderSampler(random_state=42)
            
            # Create pipeline
            pipeline = ImbPipeline([
                ('smote', smote),
                ('under', under)
            ])
            
            X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
            
            logger.info(f"Resampled from {len(X_train)} to {len(X_resampled)} samples")
            
            # Print new distribution
            unique, counts = np.unique(y_resampled, return_counts=True)
            logger.info("New class distribution:")
            for cls_idx, count in zip(unique, counts):
                cls_name = self.label_encoder.classes_[cls_idx]
                pct = (count / len(y_resampled)) * 100
                logger.info(f"  {cls_name:25s}: {count:6d} ({pct:5.1f}%)")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}, using original data")
            return X_train, y_train
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model"""
        logger.info("\n" + "="*80)
        logger.info("Training Random Forest...")
        logger.info("="*80)
        
        # Get config
        rf_config = next(
            (m for m in self.model_config["models"] if m["name"] == "random_forest"),
            {}
        )
        
        if not rf_config.get("enabled", False):
            logger.info("Random Forest disabled in config")
            return None
        
        # Train
        model = RandomForestClassifier(
            n_estimators=rf_config.get("n_estimators", 200),
            max_depth=rf_config.get("max_depth", 20),
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        
        self._print_metrics("Random Forest", y_val, y_pred, y_proba)
        
        self.models["random_forest"] = model
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        logger.info("\n" + "="*80)
        logger.info("Training XGBoost...")
        logger.info("="*80)
        
        # Get config
        xgb_config = next(
            (m for m in self.model_config["models"] if m["name"] == "xgboost"),
            {}
        )
        
        if not xgb_config.get("enabled", False):
            logger.info("XGBoost disabled in config")
            return None
        
        # Calculate class weights
        unique, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        class_weights = {cls: total / (len(unique) * count) 
                        for cls, count in zip(unique, counts)}
        
        # Map to sample weights
        sample_weights = np.array([class_weights[y] for y in y_train])
        
        # Train
        model = XGBClassifier(
            n_estimators=xgb_config.get("n_estimators", 300),
            learning_rate=xgb_config.get("learning_rate", 0.1),
            max_depth=xgb_config.get("max_depth", 10),
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        
        # Evaluate
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        
        self._print_metrics("XGBoost", y_val, y_pred, y_proba)
        
        self.models["xgboost"] = model
        return model
    
    def _print_metrics(self, model_name: str, y_true, y_pred, y_proba=None):
        """Print detailed metrics"""
        logger.info(f"\n{model_name} Results:")
        logger.info("-" * 80)
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        # Per-class metrics
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        logger.info("\nPer-Class Metrics:")
        logger.info(f"{'Class':<25s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
        logger.info("-" * 60)
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            logger.info(
                f"{class_name:<25s} "
                f"{precision[i]:>10.4f} "
                f"{recall[i]:>10.4f} "
                f"{f1[i]:>10.4f}"
            )
        
        # Check if target TPR met
        logger.info(f"\n✓ Target TPR ({self.target_tpr:.2f}) Status:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            if class_name == "normal":
                continue
            
            status = "✓" if recall[i] >= self.target_tpr else "✗"
            logger.info(f"  {status} {class_name:<25s}: {recall[i]:.4f}")
        
        # Macro averages
        logger.info(f"\nMacro Averages:")
        logger.info(f"  Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
        logger.info(f"  Recall:    {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
        logger.info(f"  F1:        {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
        
        # ROC-AUC (if probabilities available)
        if y_proba is not None and len(np.unique(y_true)) > 1:
            try:
                roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
            except:
                pass
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{cm}\n")
    
    def save_models(self, output_dir: str):
        """Save trained models and artifacts"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving models to {output_dir}...")
        
        # Save models
        for name, model in self.models.items():
            model_file = output_path / f"{name}_model.pkl"
            joblib.dump(model, model_file)
            logger.info(f"✓ Saved: {model_file}")
        
        # Save scaler
        scaler_file = output_path / "scaler.pkl"
        joblib.dump(self.scaler, scaler_file)
        logger.info(f"✓ Saved: {scaler_file}")
        
        # Save label encoder
        encoder_file = output_path / "label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_file)
        logger.info(f"✓ Saved: {encoder_file}")
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "models": list(self.models.keys()),
            "classes": list(self.label_encoder.classes_),
            "target_tpr": self.target_tpr
        }
        
        metadata_file = output_path / "metadata.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)
        
        logger.info(f"✓ Saved: {metadata_file}")
    
    def train_all(self, feature_file: str, output_dir: str):
        """Complete training pipeline"""
        
        # Load data
        df = self.load_features(feature_file)
        
        # Prepare
        X, y, feature_cols = self.prepare_data(df)
        
        # Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Scale features
        logger.info("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Balance
        X_train_balanced, y_train_balanced = self.balance_data(
            X_train_scaled, y_train
        )
        
        # Train models
        self.train_random_forest(
            X_train_balanced, y_train_balanced, X_val_scaled, y_val
        )
        
        self.train_xgboost(
            X_train_balanced, y_train_balanced, X_val_scaled, y_val
        )
        
        # Final test evaluation
        logger.info("\n" + "="*80)
        logger.info("FINAL TEST SET EVALUATION")
        logger.info("="*80)
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)
            
            self._print_metrics(f"{name.upper()} (Test Set)", y_test, y_pred, y_proba)
        
        # Save
        self.save_models(output_dir)
        
        logger.info("\n" + "="*80)
        logger.info("✓ Training complete!")
        logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--features",
        default="data/processed/features_30s.parquet",
        help="Path to feature file"
    )
    parser.add_argument(
        "--output",
        default="models",
        help="Output directory for models"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Train
    trainer = ModelTrainer(config)
    trainer.train_all(args.features, args.output)


if __name__ == "__main__":
    main()


