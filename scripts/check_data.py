"""
Data Inspection Script
Analyze generated simulation data
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from collections import Counter


def check_jsonl(filepath):
    """Check JSONL log file"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {filepath}")
    print(f"{'='*80}\n")
    
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if not records:
        print("❌ No valid records found!")
        return
    
    df = pd.DataFrame(records)
    
    print(f"📊 Basic Statistics:")
    print(f"  Total Records: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Charge points
    if "charge_point_id" in df.columns:
        cp_counts = df["charge_point_id"].value_counts()
        print(f"\n🔌 Charge Points: {len(cp_counts)}")
        print(f"  Records per CP:")
        for cp_id, count in cp_counts.head(10).items():
            print(f"    {cp_id}: {count:,}")
    
    # Anomaly labels
    if "anomaly_label" in df.columns:
        label_counts = df["anomaly_label"].value_counts()
        print(f"\n🚨 Anomaly Distribution:")
        for label, count in label_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {label:25s}: {count:7,} ({pct:5.1f}%)")
    
    # OCPP messages
    if "ocpp_message" in df.columns:
        msg_counts = df["ocpp_message"].value_counts()
        print(f"\n📨 OCPP Messages:")
        for msg, count in msg_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {msg:25s}: {count:7,} ({pct:5.1f}%)")
    
    # Numeric statistics
    numeric_cols = ["voltage", "current", "power_kw", "energy_kwh"]
    print(f"\n📈 Sensor Statistics:")
    for col in numeric_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            print(f"  {col:20s}: mean={values.mean():.2f}, "
                  f"std={values.std():.2f}, "
                  f"min={values.min():.2f}, "
                  f"max={values.max():.2f}")
    
    # Sample records
    print(f"\n📝 Sample Records (first 3):")
    for i, record in enumerate(records[:3], 1):
        print(f"\n  Record {i}:")
        for key, value in record.items():
            print(f"    {key}: {value}")
    
    print(f"\n{'='*80}\n")


def check_csv(filepath):
    """Check CSV feature file"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {filepath}")
    print(f"{'='*80}\n")
    
    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
    
    print(f"📊 Dataset Statistics:")
    print(f"  Samples: {len(df):,}")
    print(f"  Features: {len(df.columns)}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Label distribution
    if "anomaly_label" in df.columns:
        label_counts = df["anomaly_label"].value_counts()
        print(f"\n🎯 Label Distribution:")
        for label, count in label_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {label:25s}: {count:7,} ({pct:5.1f}%)")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️  Missing Values:")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            print(f"  {col:30s}: {count:6,} ({pct:5.1f}%)")
    else:
        print(f"\n✓ No missing values")
    
    # Feature types
    print(f"\n📋 Feature Types:")
    print(f"  Numeric: {len(df.select_dtypes(include=['number']).columns)}")
    print(f"  Categorical: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Sample
    print(f"\n📝 Sample Data (first 5 rows):")
    print(df.head())
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Check simulation data")
    parser.add_argument(
        "filepath",
        help="Path to data file (.jsonl, .csv, or .parquet)"
    )
    
    args = parser.parse_args()
    
    filepath = Path(args.filepath)
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return
    
    if filepath.suffix == ".jsonl":
        check_jsonl(filepath)
    elif filepath.suffix in [".csv", ".parquet"]:
        check_csv(filepath)
    else:
        print(f"❌ Unsupported file type: {filepath.suffix}")


if __name__ == "__main__":
    main()

