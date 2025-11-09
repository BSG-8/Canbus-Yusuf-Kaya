#!/usr/bin/env python3
"""
EV Charging Simulation - Main Orchestrator
Config'den okuyarak simülasyon çalıştırır
"""

import asyncio
import websockets
import subprocess
import yaml
from pathlib import Path

from src import connection_handler, run_charge_point, init_logger, get_stats, list_available_anomalies


def kill_port(port):
    """Port kullanan processleri temizle"""
    try:
        result = subprocess.run(
            ['netstat', '-ano', '-p', 'TCP'],
            capture_output=True,
            text=True
        )
        
        for line in result.stdout.splitlines():
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit():
                    print(f"[KILL] Port {port} kullanimda (PID={pid}), temizleniyor...")
                    subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
                    print(f"[OK] Port {port} temizlendi")
                    return True
        return False
    except Exception as e:
        print(f"[WARN] Port temizleme hatasi: {e}")
        return False


async def main():
    """Ana simülasyon"""
    
    # Config oku
    config_file = Path("simple_config.yaml")
    if not config_file.exists():
        print("[ERROR] simple_config.yaml bulunamadi!")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Parametreler
    host = config['csms']['host']
    port = config['csms']['port']
    num_normal = config['simulation']['num_charge_points']
    duration = config['simulation']['duration_seconds']
    
    # Anomali ayarları
    anomalies_config = config.get('anomalies', {})
    enabled_anomalies = [
        anom for anom, settings in anomalies_config.items()
        if settings.get('enabled', False)
    ]
    
    print("\n" + "="*70)
    print("EV SARJ SIMULASYONU - MODULAR SISTEM")
    print("="*70)
    print(f"CSMS: {host}:{port}")
    print(f"Simulasyon suresi: {duration}s")
    print(f"Normal istasyonlar: {num_normal}")
    print(f"Aktif anomaliler: {', '.join(enabled_anomalies) if enabled_anomalies else 'YOK'}")
    print("="*70)
    print()
    
    # Logger başlat
    init_logger(clear=True)
    
    # Port temizle
    kill_port(port)
    
    # CSMS başlat
    try:
        server = await websockets.serve(
            connection_handler,
            host,
            port,
            subprotocols=['ocpp1.6']
        )
        print(f"[OK] CSMS hazir (Port {port})\n")
    except OSError as e:
        if e.errno == 10048:
            print(f"[ERROR] Port {port} hala kullanimda!")
            print("[FIX] 5 saniye bekle...")
            await asyncio.sleep(5)
            server = await websockets.serve(
                connection_handler,
                host,
                port,
                subprotocols=['ocpp1.6']
            )
            print(f"[OK] CSMS hazir (Port {port})\n")
        else:
            raise
    
    # Charge Point'leri başlat
    tasks = []
    cp_id = 0
    
    # Normal charge points
    print(f"[START] {num_normal} normal istasyon baslatiliyor...")
    for i in range(num_normal):
        tasks.append(
            asyncio.create_task(
                run_charge_point(
                    f'CP-{cp_id:03d}',
                    host,
                    port,
                    duration=duration,
                    anomaly_type=None,
                    delay=1 + i * 0.2
                )
            )
        )
        cp_id += 1
    
    # Anomaly charge points
    for anomaly_type in enabled_anomalies:
        settings = anomalies_config[anomaly_type]
        count = settings.get('count', 1)
        anom_duration = settings.get('duration', duration)
        
        print(f"[START] {count} {anomaly_type} anomali istasyonu baslatiliyor...")
        
        for i in range(count):
            tasks.append(
                asyncio.create_task(
                    run_charge_point(
                        f'CP-{cp_id:03d}',
                        host,
                        port,
                        duration=anom_duration,
                        anomaly_type=anomaly_type,
                        delay=1 + cp_id * 0.2
                    )
                )
            )
            cp_id += 1
    
    print(f"\n[INFO] Toplam {cp_id} istasyon baslatildi. Bekleniyor...\n")
    print(f"Dashboard: http://localhost:5000\n")
    
    # Tamamlanmalarını bekle
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Sonuçlar
    stats = get_stats()
    print("\n" + "="*70)
    print("[OK] SIMULASYON TAMAMLANDI!")
    print(f"  Toplam kayit:  {stats['total']}")
    print(f"  Normal:        {stats['normal']} ({100*stats['normal']/stats['total']:.1f}%)")
    print(f"  Anomali:       {stats['anomaly']} ({stats['anomaly_rate']:.1f}%)")
    print(f"  Dosya:         {stats['file']}")
    print("="*70 + "\n")
    
    print("Mevcut anomaliler:")
    for anom in list_available_anomalies():
        print(f"  - {anom}")


if __name__ == '__main__':
    asyncio.run(main())

