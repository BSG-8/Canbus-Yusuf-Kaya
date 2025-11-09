"""
Feature Engineering Script
Extracts time-series features from raw JSONL logs for ML training
Implements windowing, statistical features, and time-domain analysis
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import yaml
from tqdm import tqdm
from scipy import stats
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts features from time-series charging data"""
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_config = config["feature_engineering"]
        self.window_sizes = self.feature_config["window_sizes"]
        
    def load_jsonl(self, filepath: str) -> pd.DataFrame:
        """Load JSONL file into DataFrame"""
        logger.info(f"Loading data from {filepath}...")
        
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
        
        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} records")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess raw data"""
        logger.info("Preprocessing data...")
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Sort by timestamp and charge point
        df = df.sort_values(["charge_point_id", "timestamp"])
        
        # Fill missing values
        numeric_cols = ["voltage", "current", "power_kw", "energy_kwh", "network_latency_ms"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col].fillna(0, inplace=True)
        
        # Remove invalid records
        initial_len = len(df)
        df = df[df["voltage"] >= 0]
        df = df[df["current"] >= 0]
        
        logger.info(f"Removed {initial_len - len(df)} invalid records")
        logger.info(f"Remaining records: {len(df)}")
        
        return df
    
    def extract_window_features(self, df: pd.DataFrame, 
                               window_size: int) -> pd.DataFrame:
        """
        Extract statistical features from sliding windows
        
        Args:
            df: Input DataFrame
            window_size: Window size in seconds
            
        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Extracting features for {window_size}s windows...")
        
        features_list = []
        
        # Group by charge point
        for cp_id, group in tqdm(df.groupby("charge_point_id"), 
                                 desc=f"Processing CPs ({window_size}s)"):
            
            group = group.sort_values("timestamp").reset_index(drop=True)
            
            # Set timestamp as index for resampling
            group_indexed = group.set_index("timestamp")
            
            # Resample to window size
            resampled = group_indexed.resample(f"{window_size}S")
            
            for timestamp, window_data in resampled:
                if len(window_data) == 0:
                    continue
                
                features = self._compute_window_features(
                    window_data, cp_id, timestamp, window_size
                )
                
                if features:
                    features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(features_df)} feature windows")
        
        return features_df
    
    def _compute_window_features(self, window: pd.DataFrame, cp_id: str, 
                                 timestamp: datetime, window_size: int) -> Dict:
        """Compute features for a single window"""
        
        if len(window) < 2:
            return None
        
        features = {
            "charge_point_id": cp_id,
            "timestamp": timestamp,
            "window_size": window_size,
            "num_samples": len(window)
        }
        
        # Get majority anomaly label in window
        if "anomaly_label" in window.columns:
            anomaly_counts = window["anomaly_label"].value_counts()
            features["anomaly_label"] = anomaly_counts.index[0]
        else:
            features["anomaly_label"] = "normal"
        
        # Statistical features for each metric
        metrics = ["voltage", "current", "power_kw", "energy_kwh"]
        
        for metric in metrics:
            if metric not in window.columns:
                continue
            
            values = window[metric].values
            
            if len(values) == 0 or np.all(np.isnan(values)):
                continue
            
            # Basic statistics
            features[f"{metric}_mean"] = np.nanmean(values)
            features[f"{metric}_std"] = np.nanstd(values)
            features[f"{metric}_min"] = np.nanmin(values)
            features[f"{metric}_max"] = np.nanmax(values)
            features[f"{metric}_median"] = np.nanmedian(values)
            
            # Range and variation
            features[f"{metric}_range"] = np.nanmax(values) - np.nanmin(values)
            features[f"{metric}_cv"] = (np.nanstd(values) / (np.nanmean(values) + 1e-10))
            
            # Percentiles
            features[f"{metric}_q25"] = np.nanpercentile(values, 25)
            features[f"{metric}_q75"] = np.nanpercentile(values, 75)
            features[f"{metric}_iqr"] = (
                np.nanpercentile(values, 75) - np.nanpercentile(values, 25)
            )
            
            # Skewness and kurtosis
            if len(values) > 3:
                features[f"{metric}_skew"] = stats.skew(values, nan_policy="omit")
                features[f"{metric}_kurtosis"] = stats.kurtosis(values, nan_policy="omit")
            
            # Delta features (change from start to end of window)
            if len(values) >= 2:
                features[f"{metric}_delta"] = values[-1] - values[0]
                features[f"{metric}_delta_pct"] = (
                    (values[-1] - values[0]) / (values[0] + 1e-10) * 100
                )
            
            # Slope (linear trend)
            if len(values) >= 3:
                x = np.arange(len(values))
                try:
                    slope, intercept = np.polyfit(x, values, 1)
                    features[f"{metric}_slope"] = slope
                except:
                    features[f"{metric}_slope"] = 0
            
            # Rate of change
            if len(values) >= 2:
                diffs = np.diff(values)
                features[f"{metric}_roc_mean"] = np.nanmean(diffs)
                features[f"{metric}_roc_std"] = np.nanstd(diffs)
                features[f"{metric}_roc_max"] = np.nanmax(np.abs(diffs))
        
        # Energy accumulation rate (should match power)
        if "power_kw" in window.columns and "energy_kwh" in window.columns:
            power_mean = features.get("power_kw_mean", 0)
            energy_delta = features.get("energy_kwh_delta", 0)
            
            # Expected energy = power * time (in hours)
            expected_energy = power_mean * (window_size / 3600.0)
            energy_error = abs(energy_delta - expected_energy)
            
            features["energy_power_consistency"] = energy_error
            features["energy_power_ratio"] = (
                energy_delta / (expected_energy + 1e-10)
            )
        
        # Power factor (voltage * current vs power)
        if all(k in window.columns for k in ["voltage", "current", "power_kw"]):
            apparent_power = window["voltage"] * window["current"] / 1000.0
            real_power = window["power_kw"]
            
            power_factor = real_power / (apparent_power + 1e-10)
            features["power_factor_mean"] = power_factor.mean()
            features["power_factor_std"] = power_factor.std()
        
        # Network latency features
        if "network_latency_ms" in window.columns:
            latency = window["network_latency_ms"].values
            features["latency_mean"] = np.nanmean(latency)
            features["latency_std"] = np.nanstd(latency)
            features["latency_max"] = np.nanmax(latency)
        
        # State encoding
        if "state" in window.columns:
            state_counts = window["state"].value_counts()
            features["state_mode"] = state_counts.index[0] if len(state_counts) > 0 else "Unknown"
        
        # Message type frequency
        if "ocpp_message" in window.columns:
            msg_counts = window["ocpp_message"].value_counts()
            features["msg_type_mode"] = msg_counts.index[0] if len(msg_counts) > 0 else "Unknown"
            features["msg_rate"] = len(window) / window_size  # messages per second
        
        return features
    
    def extract_all_features(self, input_file: str, output_dir: str):
        """Extract features for all window sizes"""
        
        # Load and preprocess
        df = self.load_jsonl(input_file)
        df = self.preprocess_data(df)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract for each window size
        for window_size in self.window_sizes:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing {window_size}s windows")
            logger.info(f"{'='*80}")
            
            features_df = self.extract_window_features(df, window_size)
            
            # Save to CSV and Parquet
            csv_file = output_path / f"features_{window_size}s.csv"
            parquet_file = output_path / f"features_{window_size}s.parquet"
            
            features_df.to_csv(csv_file, index=False)
            features_df.to_parquet(parquet_file, index=False)
            
            logger.info(f"✓ Saved: {csv_file}")
            logger.info(f"✓ Saved: {parquet_file}")
            
            # Print label distribution
            if "anomaly_label" in features_df.columns:
                label_dist = features_df["anomaly_label"].value_counts()
                logger.info("\nLabel Distribution:")
                for label, count in label_dist.items():
                    pct = (count / len(features_df)) * 100
                    logger.info(f"  {label:25s}: {count:6d} ({pct:5.1f}%)")
        
        logger.info(f"\n{'='*80}")
        logger.info("✓ Feature extraction complete!")
        logger.info(f"{'='*80}\n")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract features from simulation logs")
    parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        default="data/logs/simulation.jsonl",
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Output directory for features"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Extract features
    extractor = FeatureExtractor(config)
    extractor.extract_all_features(args.input, args.output)


if __name__ == "__main__":
    main()

