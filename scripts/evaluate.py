"""
Model Evaluation Script
Comprehensive evaluation with ROC curves, PR curves, and detailed reports
"""

import pandas as pd
import numpy as np
import logging
import yaml
import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load trained models and artifacts"""
        logger.info(f"Loading models from {self.model_dir}...")
        
        # Load scaler
        scaler_file = self.model_dir / "scaler.pkl"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            logger.info("✓ Loaded scaler")
        
        # Load label encoder
        encoder_file = self.model_dir / "label_encoder.pkl"
        if encoder_file.exists():
            self.label_encoder = joblib.load(encoder_file)
            logger.info(f"✓ Loaded label encoder: {len(self.label_encoder.classes_)} classes")
        
        # Load metadata
        metadata_file = self.model_dir / "metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"✓ Loaded metadata")
        
        # Load models
        for model_file in self.model_dir.glob("*_model.pkl"):
            model_name = model_file.stem.replace("_model", "")
            self.models[model_name] = joblib.load(model_file)
            logger.info(f"✓ Loaded model: {model_name}")
    
    def load_test_data(self, feature_file: str):
        """Load test dataset"""
        logger.info(f"Loading test data from {feature_file}...")
        
        if feature_file.endswith(".parquet"):
            df = pd.read_parquet(feature_file)
        else:
            df = pd.read_csv(feature_file)
        
        logger.info(f"Loaded {len(df)} samples")
        
        # Prepare data (same as training)
        drop_cols = ["timestamp", "charge_point_id", "anomaly_label"]
        feature_cols = [col for col in df.columns if col not in drop_cols]
        
        X = df[feature_cols].copy()
        y = df["anomaly_label"].copy()
        
        # Handle categorical
        categorical_cols = X.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            X[col] = X[col].astype("category").cat.codes
        
        # Handle missing/inf
        X.fillna(0, inplace=True)
        X.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        return X_scaled, y_encoded, y
    
    def evaluate_model(self, model_name: str, X_test, y_test, y_test_labels, 
                      output_dir: Path):
        """Comprehensive evaluation for a single model"""
        logger.info("\n" + "="*80)
        logger.info(f"Evaluating: {model_name.upper()}")
        logger.info("="*80)
        
        model = self.models[model_name]
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"\nOverall Accuracy: {accuracy:.4f}")
        
        # Per-class metrics
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            zero_division=0,
            output_dict=True
        )
        
        logger.info("\n" + classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0
        ))
        
        # Check TPR target
        target_tpr = self.metadata.get("target_tpr", 0.95) if self.metadata else 0.95
        
        logger.info(f"\n✓ TPR Target Check ({target_tpr:.0%}):")
        logger.info("-" * 60)
        
        meets_target_count = 0
        total_anomaly_classes = 0
        
        for class_name in self.label_encoder.classes_:
            if class_name == "normal":
                continue
            
            total_anomaly_classes += 1
            recall = report[class_name]["recall"]
            status = "✓ PASS" if recall >= target_tpr else "✗ FAIL"
            
            if recall >= target_tpr:
                meets_target_count += 1
            
            logger.info(f"  {status} {class_name:<25s}: Recall = {recall:.4f}")
        
        pass_rate = (meets_target_count / total_anomaly_classes * 100) if total_anomaly_classes > 0 else 0
        logger.info(f"\n  Overall Pass Rate: {meets_target_count}/{total_anomaly_classes} ({pass_rate:.1f}%)")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(report).T
        metrics_file = output_dir / f"{model_name}_metrics.csv"
        metrics_df.to_csv(metrics_file)
        logger.info(f"\n✓ Saved metrics: {metrics_file}")
        
        # Confusion Matrix
        self.plot_confusion_matrix(
            y_test, y_pred, model_name, output_dir
        )
        
        # ROC Curves
        self.plot_roc_curves(
            y_test, y_proba, model_name, output_dir
        )
        
        # PR Curves
        self.plot_pr_curves(
            y_test, y_proba, model_name, output_dir
        )
        
        return report
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, output_dir):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(14, 12))
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_file = output_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved confusion matrix: {output_file}")
    
    def plot_roc_curves(self, y_true, y_proba, model_name, output_dir):
        """Plot ROC curves for each class"""
        n_classes = len(self.label_encoder.classes_)
        
        # Convert to one-hot
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(12, 8))
        
        # Plot ROC for each class
        for i, class_name in enumerate(self.label_encoder.classes_):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
            
            plt.plot(
                fpr, tpr,
                label=f'{class_name} (AUC = {roc_auc:.3f})',
                linewidth=2
            )
        
        # Diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title(f'ROC Curves - {model_name.upper()}', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = output_dir / f"{model_name}_roc_curves.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved ROC curves: {output_file}")
    
    def plot_pr_curves(self, y_true, y_proba, model_name, output_dir):
        """Plot Precision-Recall curves"""
        n_classes = len(self.label_encoder.classes_)
        
        # Convert to one-hot
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(12, 8))
        
        # Plot PR for each class
        for i, class_name in enumerate(self.label_encoder.classes_):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_proba[:, i]
            )
            pr_auc = average_precision_score(y_true_bin[:, i], y_proba[:, i])
            
            plt.plot(
                recall, precision,
                label=f'{class_name} (AP = {pr_auc:.3f})',
                linewidth=2
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curves - {model_name.upper()}', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc='lower left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = output_dir / f"{model_name}_pr_curves.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved PR curves: {output_file}")
    
    def generate_report(self, all_reports: dict, output_dir: Path):
        """Generate comprehensive evaluation report"""
        logger.info("\n" + "="*80)
        logger.info("FINAL EVALUATION SUMMARY")
        logger.info("="*80)
        
        target_tpr = self.metadata.get("target_tpr", 0.95) if self.metadata else 0.95
        
        summary = []
        
        for model_name, report in all_reports.items():
            model_summary = {
                "model": model_name,
                "overall_accuracy": report["accuracy"],
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"]
            }
            
            # Check TPR target for each anomaly class
            anomaly_classes = [c for c in self.label_encoder.classes_ if c != "normal"]
            
            meets_target = []
            for class_name in anomaly_classes:
                recall = report[class_name]["recall"]
                meets_target.append(recall >= target_tpr)
            
            model_summary["tpr_target_pass_rate"] = (
                sum(meets_target) / len(meets_target) * 100 if meets_target else 0
            )
            
            summary.append(model_summary)
        
        summary_df = pd.DataFrame(summary)
        
        logger.info("\n" + str(summary_df.to_string(index=False)))
        
        # Save
        summary_file = output_dir / "evaluation_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"\n✓ Saved summary: {summary_file}")
        
        # Determine best model
        best_model = summary_df.loc[summary_df["tpr_target_pass_rate"].idxmax(), "model"]
        best_pass_rate = summary_df["tpr_target_pass_rate"].max()
        
        logger.info(f"\n🏆 BEST MODEL: {best_model.upper()}")
        logger.info(f"   TPR Target Pass Rate: {best_pass_rate:.1f}%")
        
        logger.info("\n" + "="*80 + "\n")
    
    def evaluate_all(self, test_data_file: str, output_dir: str):
        """Evaluate all models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load test data
        X_test, y_test, y_test_labels = self.load_test_data(test_data_file)
        
        # Evaluate each model
        all_reports = {}
        
        for model_name in self.models.keys():
            report = self.evaluate_model(
                model_name, X_test, y_test, y_test_labels, output_path
            )
            all_reports[model_name] = report
        
        # Generate summary
        self.generate_report(all_reports, output_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--models",
        default="models",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--test-data",
        default="data/processed/features_30s.parquet",
        help="Test dataset file"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for evaluation results"
    )
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.models)
    evaluator.evaluate_all(args.test_data, args.output)


if __name__ == "__main__":
    main()

