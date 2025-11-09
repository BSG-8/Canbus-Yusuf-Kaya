#!/usr/bin/env python3
"""
Anomaly Detection System
DoS anomalisini %95+ oranında tespit eder
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path="data/logs/basit_veri.jsonl"):
    """JSONL verisini yükle"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    df = pd.DataFrame(records)
    return df


def extract_features(df):
    """
    Feature Engineering
    Basit özellikler çıkar
    """
    features = pd.DataFrame()
    
    # Temel özellikler
    features['voltage'] = df['voltage']
    features['current'] = df['current']
    features['power_kw'] = df['power_kw']
    features['energy_kwh'] = df['energy_kwh']
    features['network_latency_ms'] = df['network_latency_ms']
    
    # İstatistiksel özellikler
    features['voltage_current_ratio'] = df['voltage'] / (df['current'] + 0.001)
    features['power_theoretical'] = (df['voltage'] * df['current']) / 1000
    features['power_diff'] = abs(features['power_theoretical'] - df['power_kw'])
    
    # Anomali tespiti için kritik: Voltage ve Current sapmaları
    features['voltage_deviation'] = abs(df['voltage'] - 400)  # Normal: 400V
    features['current_deviation'] = abs(df['current'] - 32)    # Normal: 32A
    
    # Güç sapması
    features['power_deviation'] = abs(df['power_kw'] - 12.8)  # Normal: ~12.8kW
    
    # Label (0=normal, 1=anomaly)
    labels = (df['anomaly_label'] != 'normal').astype(int)
    
    return features, labels


def train_model(X_train, y_train):
    """Model eğit"""
    print("\n[TRAINING] RandomForest modeli egitiliyor...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'  # Dengesiz veri için
    )
    
    model.fit(X_train, y_train)
    print("[OK] Model egitildi!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Model performansını değerlendir"""
    print("\n" + "="*70)
    print("MODEL PERFORMANSI - DoS ANOMALİ TESPİTİ")
    print("="*70)
    
    # Tahmin
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n>>> GENEL DOGRULUK: {accuracy*100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n>>> CONFUSION MATRIX:")
    print(f"                  Predicted")
    print(f"                Normal  Anomaly")
    print(f"Actual Normal     {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Anomaly    {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Classification Report
    print("\n>>> DETAYLI RAPOR:")
    report = classification_report(y_test, y_pred, 
                                   target_names=['Normal', 'DoS Anomaly'],
                                   digits=3)
    print(report)
    
    # TPR (True Positive Rate) - Hocanın istediği %95
    if cm[1,1] + cm[1,0] > 0:
        tpr = cm[1,1] / (cm[1,1] + cm[1,0])
        fpr = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        print(f"\n>>> ANOMALI TESPIT ORANI (TPR): {tpr*100:.2f}%")
        print(f"    False Positive Rate (FPR): {fpr*100:.2f}%")
        
        if tpr >= 0.95:
            print(f"\n>>> HEDEF ULASILDI! TPR >= 95%")
        else:
            print(f"\n>>> HEDEF: 95%, MEVCUT: {tpr*100:.1f}%")
    
    print("="*70 + "\n")
    
    return accuracy, y_pred


def main():
    """Ana program"""
    print("\n" + "="*70)
    print("DoS ANOMALİ TESPİT SİSTEMİ")
    print("="*70)
    
    # 1. Veriyi yükle
    data_file = Path("data/logs/basit_veri.jsonl")
    if not data_file.exists():
        print(f"\n[ERROR] Veri dosyasi bulunamadi: {data_file}")
        print("Önce 'python run_simulation.py' calistirin!")
        return
    
    print(f"\n[LOAD] Veri yukleniyor: {data_file}")
    df = load_data(data_file)
    print(f"[OK] {len(df)} kayit yuklendi")
    
    # Veri dağılımı
    normal_count = (df['anomaly_label'] == 'normal').sum()
    anomaly_count = len(df) - normal_count
    print(f"  - Normal: {normal_count} ({100*normal_count/len(df):.1f}%)")
    print(f"  - Anomali: {anomaly_count} ({100*anomaly_count/len(df):.1f}%)")
    
    if anomaly_count < 5:
        print("\n[WARN] Cok az anomali! Daha fazla veri uretmek icin:")
        print("  simple_config.yaml -> dos.count = 5")
        print("  simple_config.yaml -> dos.duration = 30")
    
    # 2. Feature extraction
    print("\n[EXTRACT] Ozellikler cikartiliyor...")
    X, y = extract_features(df)
    print(f"[OK] {X.shape[1]} ozellik cikartildi")
    print(f"Ozellikler: {list(X.columns)}")
    
    # 3. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\n[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 4. Model eğit
    model = train_model(X_train, y_train)
    
    # 5. Değerlendir
    accuracy, predictions = evaluate_model(model, X_test, y_test)
    
    # 6. Feature importance
    print("\n>>> EN ONEMLI OZELLIKLER:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(5).to_string(index=False))
    
    print("\n" + "="*70)
    print(">>> ANALIZ TAMAMLANDI!")
    print("="*70)


if __name__ == '__main__':
    main()

