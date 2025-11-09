"""
Data Logger Module
Simülasyon verilerini JSONL formatında kaydeder
"""

import json
from pathlib import Path

# Global counters
veri_sayisi = 0
anomali_sayisi = 0
veri_dosya = Path("data/logs/basit_veri.jsonl")


def init_logger(log_file=None, clear=True):
    """
    Logger'ı başlat
    
    Args:
        log_file: Log dosyası yolu (None = default)
        clear: Var olan dosyayı temizle mi?
    """
    global veri_dosya, veri_sayisi, anomali_sayisi
    
    if log_file:
        veri_dosya = Path(log_file)
    
    veri_dosya.parent.mkdir(parents=True, exist_ok=True)
    
    if clear and veri_dosya.exists():
        veri_dosya.unlink()
    
    veri_sayisi = 0
    anomali_sayisi = 0


def veri_kaydet(kayit):
    """
    Veriyi JSONL formatında kaydet
    
    Args:
        kayit: Dict olarak kayıt
    """
    global veri_sayisi, anomali_sayisi
    
    with open(veri_dosya, 'a', encoding='utf-8') as f:
        f.write(json.dumps(kayit) + '\n')
    
    veri_sayisi += 1
    if kayit.get("anomaly_label") != "normal":
        anomali_sayisi += 1


def get_stats():
    """İstatistikleri al"""
    return {
        "total": veri_sayisi,
        "normal": veri_sayisi - anomali_sayisi,
        "anomaly": anomali_sayisi,
        "anomaly_rate": (100 * anomali_sayisi / veri_sayisi) if veri_sayisi > 0 else 0,
        "file": str(veri_dosya)
    }
