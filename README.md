# 🚗⚡ EV Charging Station Simulation & Anomaly Detection

Kapsamlı bir elektrikli araç (EV) şarj istasyonu simülasyon ortamı ve anomali tespit sistemi. OCPP 1.6 protokolü üzerinden çalışan bu sistem, 9 farklı saldırı/anomali senaryosunu simüle eder ve %95+ tespit oranı hedefiyle makine öğrenmesi modelleri eğitir.

## 🎯 Özellikler

### ✅ OCPP 1.6 Simülasyonu
- **Central System (CSMS)**: WebSocket tabanlı merkezi yönetim sistemi
- **Charge Point Emülatörleri**: N adet (varsayılan 10) şarj istasyonu simülasyonu
- **Gerçekçi Metrikler**: Voltaj, akım, güç, enerji, durum kodları, heartbeat
- **Periyodik Mesajlaşma**: Heartbeat (5s), MeterValues (1s)

### 🔴 Anomali/Saldırı Senaryoları
1. **Replay Attack** - Eski mesajların yeniden gönderilmesi
2. **False Data Injection (FDI)** - Sensör değerlerinde tutarsızlık
3. **Message Tampering** - OCPP payload değişikliği
4. **Denial of Service (DoS)** - Yüksek frekanslı istek saldırısı
5. **Firmware Tampering** - Yetkisiz firmware/komut
6. **Session Hijacking** - Yetkisiz kimlikle şarj başlatma
7. **Meter Manipulation** - Sayaç değerlerini gizleme
8. **Timing Attack** - Mesaj gecikmeleri
9. **Calibration Drift** - Sensör lineer sapması

### 📊 Veri Üretimi & İşleme
- **JSONL Formatı**: Zaman serisi kayıtları
- **CSV Export**: ML eğitimi için yapılandırılmış veri
- **Özellik Mühendisliği**: 
  - Pencere tabanlı analiz (10s, 30s, 60s)
  - İstatistiksel özellikler (mean, std, min, max, slope, delta)
  - Zaman serisi özellikleri (trend, rate of change)
- **Hedef Veri Seti**: ~200-500k kayıt

### 🤖 Makine Öğrenmesi Pipeline
- **Modeller**:
  - Random Forest (özellik tabanlı)
  - XGBoost (özellik tabanlı)
  - LSTM (zaman serisi - opsiyonel)
  - Autoencoder (anomaly detection - opsiyonel)
- **Veri Dengeleme**: SMOTE oversampling
- **Değerlendirme**: ROC-AUC, PR-AUC, Confusion Matrix
- **Hedef**: TPR ≥ 95% her anomali sınıfı için

## 📁 Proje Yapısı

```
charge/
├── config.yaml                 # Ana konfigürasyon
├── requirements.txt            # Python bağımlılıkları
├── run_simulation.py           # Ana simülasyon scripti
├── src/
│   ├── central_system.py       # OCPP Central System
│   ├── charge_point_emulator.py # Charge Point emülatörü
│   ├── anomaly_injection.py    # Anomali enjeksiyon modülü
│   └── data_logger.py          # Veri loglama sistemi
├── scripts/
│   ├── generate_features.py    # Özellik çıkarma
│   ├── train_model.py          # Model eğitimi
│   └── evaluate.py             # Model değerlendirme
├── data/
│   ├── logs/                   # JSONL log dosyaları
│   ├── processed/              # CSV/Parquet veri setleri
│   └── raw/                    # Ham veri (opsiyonel)
├── models/                     # Eğitilmiş modeller
├── results/                    # Değerlendirme sonuçları
└── notebooks/                  # Jupyter notebook'lar (opsiyonel)
```

## 🚀 Kurulum

### 1. Gereksinimler
- Python 3.10 veya üzeri
- Windows/Linux/macOS

### 2. Sanal Ortam Oluşturma
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleme
```bash
pip install -r requirements.txt
```

## 📖 Kullanım

### Adım 1: Simülasyonu Çalıştırma

```bash
python run_simulation.py
```

**Çıktı:**
- `data/logs/simulation.jsonl` - Raw OCPP mesajları ve metrikler
- `data/logs/simulation_debug.log` - Debug logları

**Simülasyon Parametreleri** (`config.yaml`):
```yaml
simulation:
  duration_seconds: 3600  # 1 saat
  charge_points: 10       # 10 şarj noktası
  heartbeat_interval: 5
  meter_values_interval: 1
```

### Adım 2: Özellikleri Çıkarma

```bash
python scripts/generate_features.py
```

**Çıktı:**
- `data/processed/features_10s.csv` / `.parquet`
- `data/processed/features_30s.csv` / `.parquet`
- `data/processed/features_60s.csv` / `.parquet`

### Adım 3: Model Eğitimi

```bash
python scripts/train_model.py --features data/processed/features_30s.parquet
```

**Çıktı:**
- `models/random_forest_model.pkl`
- `models/xgboost_model.pkl`
- `models/scaler.pkl`
- `models/label_encoder.pkl`

### Adım 4: Model Değerlendirme

```bash
python scripts/evaluate.py --test-data data/processed/features_30s.parquet
```

**Çıktı:**
- `results/xgboost_confusion_matrix.png`
- `results/xgboost_roc_curves.png`
- `results/xgboost_pr_curves.png`
- `results/evaluation_summary.csv`

## ⚙️ Konfigürasyon

### Anomali Senaryolarını Özelleştirme

`config.yaml` dosyasında her anomali için parametreler:

```yaml
anomalies:
  false_data_injection:
    enabled: true
    start_time: 600          # Saniye
    duration: 240            # Saniye
    target_cps: [3, 4]       # Hedef charge point'ler
    voltage_change_percent: 20
    current_change_percent: 30
```

### Model Parametreleri

```yaml
model_training:
  target_tpr: 0.95           # %95 minimum recall
  balance_classes: true
  use_smote: true
  
  models:
    - name: "xgboost"
      enabled: true
      n_estimators: 300
      learning_rate: 0.1
      max_depth: 10
```

## 📊 Örnek Veri Formatı

### JSONL Log Kaydı
```json
{
  "timestamp": "2025-11-09T17:00:01Z",
  "charge_point_id": "CP-001",
  "transaction_id": "T-1234",
  "voltage": 400.2,
  "current": 32.1,
  "power_kw": 12.9,
  "energy_kwh": 1.234,
  "state": "Charging",
  "ocpp_message": "MeterValues",
  "network_latency_ms": 25,
  "anomaly_label": "normal"
}
```

## 📈 Beklenen Sonuçlar

### Veri Dağılımı
- **Normal**: %70-80
- **Her Anomali**: %2-5 (toplam 9 anomali tipi)
- **Toplam Kayıt**: 200,000 - 500,000

### Model Performansı Hedefi
- **TPR (Recall)**: ≥ %95 her anomali sınıfı için
- **FPR**: ≤ %5
- **ROC-AUC**: ≥ 0.90

## 🔍 Anomali Detay Açıklamaları

### 1. Replay Attack
**Açıklama**: Saldırgan eski OCPP mesajlarını kaydedip tekrar gönderiyor.

**Parametreler**:
- `replay_delay`: Kaç saniye önceki mesajları tekrar göndereceği
- `frequency`: Saniyede kaç mesaj

**Tespit**: Zaman damgası tutarsızlığı, tekrar eden transaction ID'ler

### 2. False Data Injection (FDI)
**Açıklama**: Sensör değerlerinde yapay değişiklik.

**Parametreler**:
- `voltage_change_percent`: Voltaj sapma yüzdesi
- `current_change_percent`: Akım sapma yüzdesi

**Tespit**: İstatistiksel aykırı değerler, güç-enerji tutarsızlığı

### 3. DoS Attack
**Açıklama**: Sistemi yüksek frekanslı isteklerle boğma.

**Parametreler**:
- `heartbeat_frequency`: Saniyede mesaj sayısı
- `connection_flood`: Bağlantı saldırısı

**Tespit**: Anormal mesaj frekansı, yüksek ağ trafiği

## 🛠️ Geliştirme

### Test Etme
```bash
# Tek charge point testi
python src/charge_point_emulator.py

# Central system testi
python src/central_system.py
```

### Loglama Seviyeleri
`run_simulation.py` içinde:
```python
logging.basicConfig(level=logging.DEBUG)  # Detaylı loglar
logging.basicConfig(level=logging.INFO)   # Normal
```

## 📚 Teknik Detaylar

### OCPP Mesaj Akışı
1. **BootNotification**: Başlangıç kaydı
2. **Heartbeat**: Her 5 saniyede
3. **Authorize**: ID tag doğrulama
4. **StartTransaction**: Şarj başlangıcı
5. **MeterValues**: Her 1 saniyede metrik gönderimi
6. **StopTransaction**: Şarj bitişi

### Özellik Mühendisliği
- **Pencere Boyutu**: 10s, 30s, 60s
- **İstatistiksel**: mean, std, min, max, median, Q25, Q75, IQR
- **Trend**: slope, delta, rate of change
- **Domain-specific**: energy-power consistency, power factor

## 🤝 Katkıda Bulunma

Proje bir ders projesi olarak geliştirilmiştir. İyileştirme önerileri:

1. OCPP 2.0.1 desteği
2. Gerçek zamanlı stream işleme (Kafka)
3. Deep Learning modelleri (Transformer, CNN-LSTM)
4. Web dashboard (Grafana)

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 📧 İletişim

Proje hakkında sorularınız için lütfen proje sahibiyle iletişime geçin.

---

## 🎯 Hızlı Başlangıç (TL;DR)

```bash
# 1. Kurulum
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Simülasyon (1 saat)
python run_simulation.py

# 3. Özellik çıkarma
python scripts/generate_features.py

# 4. Model eğitimi
python scripts/train_model.py

# 5. Değerlendirme
python scripts/evaluate.py

# Sonuçlar: results/ klasöründe
```

## ✨ Öne Çıkan Özellikler

✅ **Tam OCPP 1.6 uyumlu** simülasyon  
✅ **9 farklı anomali senaryosu** ile kapsamlı test  
✅ **Parametrik konfigürasyon** - tüm ayarlar config.yaml'da  
✅ **Otomatik veri üretimi** - 200k+ etiketli kayıt  
✅ **Production-ready ML pipeline** - SMOTE, cross-validation, hyperparameter tuning  
✅ **Detaylı görselleştirme** - ROC, PR curves, confusion matrix  
✅ **%95+ tespit oranı hedefi** - her anomali sınıfı için  

---

**Not**: İlk çalıştırmada simülasyon 1 saat sürebilir. Test için `config.yaml`'da `duration_seconds` değerini azaltabilirsiniz (örn. 300 = 5 dakika).

