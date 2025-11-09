"""
Anomaly Injection Module
Hocanın DoS senaryosu: "Operasyonel Felç"
"""

import random
import asyncio


# ============ DoS ATTACK (Operasyonel Felç) ============

def dos_attack_injector(cp_id):
    """
    DoS Attack: "Operasyonel Felç"
    
    Hocanın senaryosu:
    - Vektör A: RemoteStopTransaction flood (aktif şarjları durdur)
    - Vektör B: BootNotification flood (CSMS'i kilitle)
    
    Bu simülasyonda: Her 3 saniyede çok hızlı mesajlar (flood)
    """
    def inject(second, voltage, current, power, energy):
        # Her 3 saniyede bir anormal spike (flood simülasyonu)
        if second % 3 == 0:
            # Aşırı yüksek değerler = flood etkisi
            voltage = random.uniform(350, 550)
            current = random.uniform(40, 90)
            power = (voltage * current) / 1000
        return voltage, current, power, energy
    return inject


# ============ ANOMALİ HARITASI (SADECE DoS) ============

ANOMALY_MAP = {
    "dos": dos_attack_injector,
    "dos_attack": dos_attack_injector,
}


def get_anomaly_injector(anomaly_type, cp_id):
    """
    Anomali tipi için enjektör fonksiyonu döndür
    
    Args:
        anomaly_type: Anomali tipi (None, "dos", "voltage_spike", vs.)
        cp_id: Charge point ID
    
    Returns:
        Anomali enjektör fonksiyonu veya None
    """
    if not anomaly_type or anomaly_type == "normal":
        return None
    
    injector_factory = ANOMALY_MAP.get(anomaly_type.lower())
    if injector_factory:
        return injector_factory(cp_id)
    
    print(f"[WARN] Bilinmeyen anomali tipi: {anomaly_type}")
    return None


def list_available_anomalies():
    """Mevcut anomali türlerini listele"""
    return ["dos (Operasyonel Felç - DoS Attack)"]
